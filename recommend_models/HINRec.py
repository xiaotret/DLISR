import os
from core import data_repository
from Helpers.util import save_dict, read_dict, sigmoid
from text_utils.word_embedding import get_embedding_matrix
from core.evalute import evalute_by_epoch
from Helpers.HIN_sim import mashup_HIN_sims
from text_utils.gensim_data import get_default_gd

from random import choice
import numpy as np
import math


# 该模型在训练时，训练样本的相似度文件临时存储在self.mID2PathSims中，不需永久存储，一次训练即可
# 为NI服务时，训练和测试样本的相似度文件都需要永久存储；包括各种path的相似度和某一种权重组合的相似度，以及特征

class HINRec(object):
    def __init__(self,args,model_name = 'PasRec',semantic_mode='HDP',LDA_topic_num='',epoch_num=15,neighbor_size=15,topTopicNum=3,cluster_mode='LDA',cluster_mode_topic_num=100):
        # semantic_mode='HDP',LDA_topic_num=None: about feature in HIN
        # cluster_mode='LDA',cluster_mode_topic_num: ABOUT clustering by LDA...

        self.simple_name = model_name
        if self.simple_name == 'IsRec_best':
            self.p1_weight, self.p2_weight, self.p3_weight = 1/3,1/3,1/3
            self.path_weights = [self.p1_weight, self.p2_weight, self.p3_weight]
        elif self.simple_name == 'PasRec_2path':
            self.p1_weight, self.p2_weight = 1/2,1/2
            self.path_weights = [self.p1_weight, self.p2_weight]
        elif self.simple_name == 'IsRec':
            self.p1_weight,self.p2_weight,self.p3_weight,self.p4_weight,self.p5_weight,self.p6_weight,self.p7_weight = 1/7,1/7,1/7,1/7,1/7,1/7,1/7
            self.path_weights = [self.p1_weight,self.p2_weight,self.p3_weight,self.p4_weight,self.p5_weight,self.p6_weight,self.p7_weight]
        else :
            self.p1_weight,self.p2_weight,self.p3_weight,self.p4_weight,self.p5_weight,self.p6_weight = 1/6,1/6,1/6,1/6,1/6,1/6
            self.path_weights = [self.p1_weight,self.p2_weight,self.p3_weight,self.p4_weight,self.p5_weight,self.p6_weight]

        self.neighbor_size = neighbor_size # 找最近邻时的规模
        self.epoch_num = epoch_num
        self.learning_rate = 0.001
        self.reg=0.001
        self.sample_ratio = 50 # pairwise优化，每个api对应的训练pair数目

        self.model_name = '{}_{}_epoch{}_nbSize{}TopicNum{}{}{}'.format(self.simple_name,semantic_mode, epoch_num, neighbor_size,topTopicNum,cluster_mode,cluster_mode_topic_num)
        self.model_dir = data_repository.get_ds().model_path.format(self.model_name) # 模型路径 # !!!
        self.weight_path = os.path.join(self.model_dir, 'weights.npy')  # 最核心的数据，只保存它，其他无用！

        # 训练数据集 api_id: set(mashup_ids)
        self.train_aid2mids = {}
        for mashup_id, api_id in data_repository.get_ds().train_mashup_api_list:
            if api_id not in self.train_aid2mids.keys():
                self.train_aid2mids[api_id] = set()
            self.train_aid2mids[api_id].add(mashup_id)
        self.his_a_ids = list(self.train_aid2mids.keys())  # 训练数据集中出现的api_id !!!
        self.notInvokeScore = 0 # 加入评价的api是历史mashup从未调用过的，基准评分0；参考1和0  0.5很差!!!

        # 文本，HIN相似度相关
        self.HIN_path = os.path.join(self.model_dir, 'HIN_sims') # 存储各个HIN_sim源文件的root !!!
        self.semantic_mode = semantic_mode
        self.LDA_topic_num = LDA_topic_num

        # HIN中 文本相似度计算  只在IsRec_best中使用，因为PasRec和IsRec计算文本相似度时要么使用topic作为tag，要么使用EmbMax!!!
        HIN_gd = get_default_gd(tag_times=2,strict_train=False)
        embedding_matrix = get_embedding_matrix(HIN_gd.dct.token2id, args.embedding_name,dimension=args.embedding_dim) # 每个编码词的embedding
        HIN_gd.model_pcs(model_name = self.semantic_mode,LDA_topic_num =self.LDA_topic_num) # IsRec_best需要使用TF_IDF
        HIN_gd.get_all_encoded_comments()
        self.mhs = mashup_HIN_sims(embedding_matrix, gd = HIN_gd, semantic_name=self.semantic_mode,
                                   HIN_path=self.HIN_path,features=(HIN_gd._mashup_features, HIN_gd._api_features),
                                   if_text_sem=True,if_tag_sem=False)
        self.mID2PathSims={} # 每个mashupID(含已调用apis)，跟历史mashup的各种路径的相似度
        self.HIN_sims_changed_flag = False

        # topTopicNum在PasRec中用于基于LDA等的主题计算content相似度；在IsRec中用于从K个类中寻找近邻!!!
        self.topTopicNum = topTopicNum
        topic_gd = get_default_gd(tag_times=0,strict_train=True) # 用gensim处理文本,文本中不加tag
        topic_gd.model_pcs(model_name = cluster_mode,LDA_topic_num =cluster_mode_topic_num) # 暂时用HDP分类/提取特征;确定主题数之后改成LDA
        self.m_id2topic,self.a_id2topic = topic_gd.get_topTopics(self.topTopicNum)
        # 全部mashup: topic到mashup的映射；相当于按主题分类
        self.topic2m_ids = {}
        for m_id,topic_indexes in enumerate(self.m_id2topic):
            for topic_index in topic_indexes:
                if topic_index not in self.topic2m_ids:
                    self.topic2m_ids[topic_index] = []
                self.topic2m_ids[topic_index].append(m_id)

        self.read_model() # 主要读取权重参数，其他不重要

    # 计算一个mashup(可能需要已选择服务)到其他mashup的各种相似度，可选择是否存入到self.mID2PathSims中(NI的实例调用时)
    def get_id2PathSims(self,m_id,slt_apis_list=None,if_temp_save=True,if_cutByTopics=True):
        key = (m_id,tuple(slt_apis_list)) if slt_apis_list else m_id
        if key in self.mID2PathSims.keys(): # 重新加载该模型时为空，有必要时为NI即时计算
            return self.mID2PathSims.get(key)
        else:
            his_m_ids = set(data_repository.get_ds().his_mashup_ids)-set([m_id])
            if 'IsRec' in self.simple_name and if_cutByTopics:
                # IsRec是否使用剪枝策略：拥有相同tag的所有mashup中选择近邻
                final_his_m_ids = []
                for topic in self.m_id2topic[m_id]:
                    final_his_m_ids += list(filter(lambda x: x in his_m_ids,self.topic2m_ids[topic]))
                his_m_ids = final_his_m_ids

            if self.simple_name == 'PasRec':
                id2P1Sim = {neigh_m_id: self.mhs.get_p1_sim(min(m_id, neigh_m_id), max(m_id, neigh_m_id), 'MetaPath') for neigh_m_id in his_m_ids }
                # 特殊：计算文本相似度时，使用content的topic作为tag，用get_p1_sim
                id2P2Sim = {neigh_m_id: self.mhs.get_p1_sim(min(m_id, neigh_m_id), max(m_id, neigh_m_id), 'MetaPath', self.m_id2topic) for neigh_m_id in his_m_ids }
                id2P3Sim = {neigh_m_id: self.mhs.get_p3_sim(neigh_m_id, slt_apis_list) for neigh_m_id in his_m_ids }
                id2P4Sim = {neigh_m_id: self.mhs.get_p4_sim(neigh_m_id, slt_apis_list, 'MetaPath') for neigh_m_id in his_m_ids}
                # 特殊：计算文本相似度时，使用content的topic作为tag，用get_p4_sim
                id2P5Sim = {neigh_m_id: self.mhs.get_p4_sim(neigh_m_id, slt_apis_list, 'MetaPath', self.a_id2topic) for neigh_m_id in his_m_ids}
                id2P6Sim = {neigh_m_id: self.mhs.get_p6_sim(neigh_m_id, slt_apis_list) for neigh_m_id in his_m_ids}
                id2PathSims = [id2P1Sim, id2P2Sim, id2P3Sim, id2P4Sim, id2P5Sim, id2P6Sim]  #
            elif self.simple_name == 'IsRec':
                id2P1Sim = {neigh_m_id: self.mhs.get_p1_sim(min(m_id, neigh_m_id), max(m_id, neigh_m_id), 'MetaPath') for neigh_m_id in his_m_ids}
                id2P2Sim = {neigh_m_id: self.mhs.get_p2_sim(min(m_id, neigh_m_id), max(m_id, neigh_m_id), 'EmbMax') for neigh_m_id in his_m_ids}
                id2P3Sim = {neigh_m_id: self.mhs.get_p3_sim(neigh_m_id, slt_apis_list) for neigh_m_id in his_m_ids}
                id2P4Sim = {neigh_m_id: self.mhs.get_p4_sim(neigh_m_id, slt_apis_list, 'MetaPath') for neigh_m_id in  his_m_ids}
                id2P5Sim = {neigh_m_id: self.mhs.get_p5_sim(neigh_m_id, slt_apis_list, 'EmbMax') for neigh_m_id in his_m_ids}
                id2P6Sim = {neigh_m_id: self.mhs.get_p6_sim(neigh_m_id, slt_apis_list) for neigh_m_id in his_m_ids}
                id2P7Sim = {neigh_m_id: self.mhs.get_p2_sim_sem(min(m_id, neigh_m_id), max(m_id, neigh_m_id), 'TF_IDF') for neigh_m_id in his_m_ids}
                id2PathSims = [id2P1Sim, id2P2Sim, id2P3Sim, id2P4Sim, id2P5Sim, id2P6Sim,id2P7Sim]  #
            elif self.simple_name == 'PasRec_2path':
                id2P1Sim = {neigh_m_id: self.mhs.get_p1_sim(min(m_id, neigh_m_id), max(m_id, neigh_m_id), 'MetaPath') for neigh_m_id in his_m_ids}
                id2P2Sim = {neigh_m_id: self.mhs.get_p1_sim(min(m_id, neigh_m_id), max(m_id, neigh_m_id), 'MetaPath', self.m_id2topic) for neigh_m_id in his_m_ids}
                id2PathSims = [id2P1Sim, id2P2Sim]  #
            elif self.simple_name == 'IsRec_best':
                id2P1Sim = {neigh_m_id: self.mhs.get_p1_sim(min(m_id, neigh_m_id), max(m_id, neigh_m_id), 'MetaPath') for neigh_m_id in his_m_ids}
                id2P2Sim = {neigh_m_id: self.mhs.get_p2_sim(min(m_id, neigh_m_id), max(m_id, neigh_m_id), 'EmbMax') for neigh_m_id in his_m_ids}
                id2P3Sim = {neigh_m_id: self.mhs.get_p2_sim_sem(min(m_id, neigh_m_id), max(m_id, neigh_m_id), 'TF_IDF') for neigh_m_id in his_m_ids}
                id2PathSims = [id2P1Sim, id2P2Sim, id2P3Sim]  #
            if if_temp_save:
                self.mID2PathSims[key] = id2PathSims
            return id2PathSims

    def predict_an_instance(self,m_id,a_id,slt_apis_list,if_score_only = False):
        # 通用:根据基于路径的相似度，预测当前mashup(在已选择slt_apis_list的情况下)对api的一种评分
        def get_path_score(m_id, a_id, neighborId2sim):
            if a_id not in self.train_aid2mids.keys(): # 没有被调用过的服务，user-based的机制，评分肯定为0
                score_sum = self.notInvokeScore
            else:
                # 计算某种路径下的score/sim，要输入一个mashup和每个历史mashup的某种路径下的相似度
                num= min(self.neighbor_size, len(neighborId2sim))
                sorted_id2sim = sorted(neighborId2sim.items(), key=lambda x:x[1], reverse=True) [:num]
                # 最相似的近邻 论文中提到还需要是调用过api的mashup，这里通过分值为0可以达到同样目的
                neighbor_m_ids,sims = zip(*sorted_id2sim)
                temp_scores = [1 if m_id in self.train_aid2mids[a_id] else 0 for m_id in neighbor_m_ids]
                score_sum = sum(np.array(temp_scores)*np.array(sims))
            return score_sum

        id2PathSims = self.get_id2PathSims(m_id,slt_apis_list) # 计算某个实例的pathSims，跟weight无关，计算一次不会变
        path_scores = [get_path_score(m_id,a_id,id2PathSim) for id2PathSim in id2PathSims] # 几种路径下的score,更新模型时有用
        score = sum(np.array(path_scores)*np.array(self.path_weights))
        if if_score_only:
            return score
        else:
            return score,path_scores

    def get_true_candi_apis(self):
        # TODO: 没用到
        # 根据IsRec的思想，只把近邻mashup调用过的服务作为候选
        self.mid2candiAids = None
        self.mid2candiAids_path = os.path.join(self.model_dir, 'true_candi_apis.txt')

        if not os.path.exists(self.mid2candiAids_path):
            for key,id2PathSims in self.mID2PathSims.items():
                m_id = key[0] # key = (m_id,tuple(slt_apis_list))
                if m_id not in self.mid2candiAids.keys():
                    all_neighbor_mids = set()
                    for id2sim in id2PathSims: # 到各个剪枝后的候选近邻的某种路径下的相似度
                        num= min(self.neighbor_size,len(id2sim))
                        sorted_id2sim = sorted(id2sim.items(),key=lambda x:x[1],reverse=True) [:num] # 某种路径下的近邻
                        sorted_ids,_ = zip(*sorted_id2sim)
                        all_neighbor_mids = all_neighbor_mids.union(set(sorted_ids))
                    true_candi_apis = set()
                    for neighbor_mid in all_neighbor_mids:
                        if neighbor_mid in data_repository.get_ds().train_mashup_api_dict.keys():
                            true_candi_apis = true_candi_apis.union(set(data_repository.get_ds().train_mashup_api_dict[neighbor_mid])) # 该近邻mashup调用过的api
                    self.mid2candiAids[m_id] = true_candi_apis
            save_dict(self.mid2candiAids_path,self.mid2candiAids)
        else:
            self.mid2candiAids = read_dict(self.mid2candiAids_path)
        return self.mid2candiAids

    def predict(self,args): # 仿照DL的model,返回多个实例的评分
        m_ids, a_ids, slt_apis_lists = args[0],args[1],args[2]
        num = len(m_ids)
        if not slt_apis_lists:
            predictions = [self.predict_an_instance(m_ids[i], a_ids[i], None, if_score_only=True) for i in range(num)]
        else:
            predictions = [self.predict_an_instance(m_ids[i], a_ids[i], slt_apis_lists[i],if_score_only = True) for i in range(num)]
        return np.array(predictions)

    def get_instances(self,instances_dict):
        # 不需要array，不需要填充，输出结果可以使用predict就好;
        # 供evalute_by_epoch使用
        return instances_dict.get('mashup'), instances_dict.get('api'), instances_dict.get('slt_apis',None)

    def train(self,test_data):
        """
        模仿librec的实现，每个api跟一对正负mashup组成一个样例，每个api的样本数最大为50；(均衡性问题？）
        每20次测试一次，训练数据不用输入，用dataset
        :param test_data:
        :return:
        """
        for index in range(self.epoch_num):
            loss = 0
            for sampleCount in range(len(self.his_a_ids) * self.sample_ratio):  # 每个
                while True:
                    a_id = choice(self.his_a_ids)
                    if len(self.train_aid2mids[a_id]) == len(data_repository.get_ds().his_mashup_ids):  # 如果被所有mashup调用，则没有负例
                        continue
                    pos_m_ids = self.train_aid2mids[a_id]  # 正例
                    pos_m_id = choice(list(pos_m_ids))
                    neg_m_ids = data_repository.get_ds().his_mashup_ids_set - pos_m_ids
                    neg_m_id = choice(list(neg_m_ids))
                    break

                # 训练时计算相似度，已选择的服务应该不包含当前服务
                posPredictRating,posPathScores = self.predict_an_instance(pos_m_id, a_id, data_repository.get_ds().train_mashup_api_dict[pos_m_id]-{a_id})
                negPredictRating,negPathScores = self.predict_an_instance(neg_m_id, a_id, data_repository.get_ds().train_mashup_api_dict[neg_m_id]-{a_id})
                diffValue = posPredictRating - negPredictRating
                deriValue = sigmoid(-diffValue);
                lossValue = -math.log(sigmoid(diffValue))
                loss += lossValue

                for i in range(len(self.path_weights)): # 优化第i条路径对应的权重参数
                    temp_value = self.path_weights[i]
                    self.path_weights[i] += self.learning_rate * (deriValue * (posPathScores[i]-negPathScores[i]) - self.reg * temp_value)
                    loss += self.reg * temp_value * temp_value
            print('epoch:{}, loss:{}'.format(index, loss))

            if index>0 and index%20==0:
                self.test_model(test_data)

    def test_model(self,test_data):
        evalute_by_epoch(self, self, self.model_name, test_data)

    def save_model(self):
        # 存储HIN相似度文件和参数权重
        np.save(self.weight_path ,np.array(self.path_weights))
        print('save weights,done!')

    def read_model(self):
        if os.path.exists(self.weight_path):
            self.path_weights = np.load(self.weight_path)
            print('read weights,done!')

