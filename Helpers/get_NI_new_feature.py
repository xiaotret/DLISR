import os
import pickle
import numpy as np

from core.dataset import dataset
from core import data_repository

class get_NI_new_feature(object):
    def __init__(self,args):
        self.args = args
        self.NI_sim_mode = args.NI_sim_mode
        self.path_topK_mode = args.path_topK_mode
        self.topK = args.topK
        pass

    def process(self, sim_model=None, train_data=None, test_data=None):
        # 准备各种相似度：可以是提供文本和tag特征的CI，也可以是提供相似度支持的HINRec_model
        self.his_mashup_NI_feas = data_repository.get_ds().MF_obj.mashup_emb_df['embedding'][data_repository.get_ds().his_mashup_ids].tolist() # TODO
        if isinstance(self.his_mashup_NI_feas[0],str):
            self.his_mashup_NI_feas = list(map(eval,self.his_mashup_NI_feas))
        self.his_mashup_NI_feas = np.array(self.his_mashup_NI_feas)
        if self.NI_sim_mode == 'tagSim': # 基于CI部分的特征计算相似度，MISR使用 TODO
            self.set_mashup_api_features(sim_model)
        else:
            self.m2neighors_path = os.path.join(sim_model.model_dir, 'm2neighors.dat')
            self.m2neighors = {}
            self.path_weights = sim_model.path_weights # 读取预训练的相似度模型中的meta-path权重

            self.m2AllSimsPath = os.path.join(sim_model.model_dir, 'mID2AllSims_{}.sim'.format(self.NI_sim_mode))
            self.m2ASimPath = os.path.join(sim_model.model_dir, 'mID2ASim_{}_{}_{}.sim'.format(self.NI_sim_mode, self.path_topK_mode, self.topK))
            self.m2ASim, self.m2AllSims = {}, {}

            all_paths_sim_modes = ['PasRec','PasRec_2path','IsRec','IsRec_best']
            if self.NI_sim_mode in all_paths_sim_modes: # 计算mashup表示时需要已选择服务()
                self.m2NI_feas = {}
                self.m2NI_feas_path = os.path.join(sim_model.model_dir, 'NI_m_id2{}_{}_{}.feas'.format(self.NI_sim_mode, self.path_topK_mode, self.topK))
                self.get_samples_m_feas(train_data, test_data,sim_model)

    def get_m2AllSims(self, train_data, test_data, sim_model):
        # 读取mid/(mid,slt_aids)到每个历史mashup的相似度（IsRec的已经剪枝，只有一部分mashup）
        if os.path.exists(self.m2AllSimsPath):  # !!! 为了显式不得已，再改  self.m2neighors[key]
            with open(self.m2AllSimsPath, 'rb') as f:
                self.m2AllSims = pickle.load(f)
        else:
            # 先求self.m2AllSims，后存
            mashup_id_list = train_data.get('mashup')
            api_id_list = train_data.get('api')
            mashup_slt_apis_list = train_data.get('slt_apis')

            for i in range(len(mashup_id_list)):
                if self.NI_sim_mode == 'PasRec' or self.NI_sim_mode == 'IsRec':  # 需要已选择的服务
                    key = (mashup_id_list[i], tuple(mashup_slt_apis_list[i]))
                    if key not in self.m2AllSims.keys():
                        self.m2AllSims[key] = sim_model.get_id2PathSims(*key, if_temp_save=False)
                else:
                    key = mashup_id_list[i]
                    if key not in self.m2AllSims.keys():
                        self.m2AllSims[key] = sim_model.get_id2PathSims(key, if_temp_save=False)
            print('compute id2AllPathSims for train_data,done!')

            mashup_id_list = test_data.get('mashup')
            api_id_list = test_data.get('api')
            mashup_slt_apis_list = test_data.get('slt_apis')

            for i in range(len(mashup_id_list)):
                for j in range(len(api_id_list[i])):
                    if self.NI_sim_mode == 'PasRec' or self.NI_sim_mode == 'IsRec':  # 需要已选择的服务
                        key = (mashup_id_list[i][j], tuple(mashup_slt_apis_list[i]))
                        if key not in self.m2AllSims.keys():
                            self.m2AllSims[key] = sim_model.get_id2PathSims(*key, if_temp_save=False)
                    else:
                        key = mashup_id_list[i][j]
                        if key not in self.m2AllSims.keys():
                            self.m2AllSims[key] = sim_model.get_id2PathSims(key, if_temp_save=False)
            print('compute id2AllPathSims for test_data,done!')

            with open(self.m2AllSimsPath, 'wb') as f:
                pickle.dump(self.m2AllSims, f)
        return self.m2AllSims

    def get_m2ASim(self, train_data, test_data, sim_model):
        """得到一个mashup到其他mashup的归一化的综合相似度向量"""
        if os.path.exists(self.m2neighors_path) and os.path.exists(self.m2ASimPath):  # ...计算explicit用
            with open(self.m2ASimPath, 'rb') as f:
                self.m2ASim = pickle.load(f)
            with open(self.m2neighors_path, 'rb') as f:
                self.m2neighors = pickle.load(f)
        else:  # 一次性计算全部的并存储
            print('m2ASim not exist, computing!')
            dict_ = self.get_m2AllSims(train_data, test_data, sim_model)  # self.m2AllSims 每个sample的相似度映射
            for key, id2PathSims in dict_.items():
                m_id = key if isinstance(key, int) else key[0]  # mashup ID

                if self.path_topK_mode == 'eachPathTopK':  # 每个路径的topK
                    for i in range(len(id2PathSims)):  # 某一种路径的相似度
                        id2PathSim = id2PathSims[i]
                        num = min(self.topK, len(id2PathSim))
                        id2PathSim = sorted(id2PathSim.items(), key=lambda x: x[1], reverse=True)[:num]
                        id2PathSims[i] = {key: value for key, value in id2PathSim}

                id2score = {his_m_id: 0 for his_m_id in data_repository.get_ds().his_mashup_ids}  # 到所有历史mashup的综合相似度
                for his_m_id in id2score.keys():  # 每个历史近邻mashup
                    if his_m_id != m_id:  # 除去自身
                        for path_index, id2aPathSim in enumerate(id2PathSims):  # 每种相似度路径
                            pathSim = 0 if his_m_id not in id2aPathSim.keys() else id2aPathSim[his_m_id]  # 某个历史mid可能没有某种相似度
                            id2score[his_m_id] += pathSim * self.path_weights[path_index]

                # 为显式设计，综合所有路径之后存储topk近邻
                num = min(self.topK, len(id2score))
                self.m2neighors[key], _ = zip(
                    *(sorted(id2score.items(), key=lambda x: x[1], reverse=True)[:num]))  # 按顺序存储topK个近邻的ID

                if self.path_topK_mode == 'allPathsTopK':  # 最终所有路径综合评分的topK
                    num = min(self.topK, len(id2score))
                    id2score = sorted(id2score.items(), key=lambda x: x[1], reverse=True)[:num]
                    id2score = {key: value for key, value in id2score}

                sims = np.array([id2score[his_m_id] if his_m_id in id2score.keys() else 0 for his_m_id in
                                 data_repository.get_ds().his_mashup_ids])  # 按顺序排好的sims: (#his_m_ids)
                sum_sim = sum(sims)
                if sum_sim == 0:
                    print('sims sum=0!')
                else:
                    sims = sims / sum_sim
                self.m2ASim[key] = sims

            print('m2ASim, computed!')
            with open(self.m2ASimPath, 'wb') as f:
                pickle.dump(self.m2ASim, f)

            with open(self.m2neighors_path, 'wb') as f:
                pickle.dump(self.m2neighors, f)
        return self.m2ASim

    def get_samples_m_feas(self, train_data, test_data, sim_model):
        if os.path.exists(self.m2neighors_path) and os.path.exists(self.m2NI_feas_path):  # 一步到位，读取特征
            with open(self.m2NI_feas_path, 'rb') as f:
                self.m2NI_feas = pickle.load(f)
            with open(self.m2neighors_path, 'rb') as f:
                self.m2neighors = pickle.load(f)
        else:
            print('m2NI_feas not exist, computing!')
            dict_ = self.get_m2ASim(train_data, test_data, sim_model)  # 综合sim的映射 self.m2ASim
            print('compute m2NI_feas,done!')
            # 获得self.id2NI_sims之后即时计算全部样本的NI feature
            for key, value in dict_.items():
                self.m2NI_feas[key] = np.dot(value, self.his_mashup_NI_feas)

            with open(self.m2NI_feas_path, 'wb') as f:
                pickle.dump(self.m2NI_feas, f)