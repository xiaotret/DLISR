import heapq
import os
import sys
import numpy as np

from core import data_repository
from path_config import evaluate_path
from text_utils.gensim_data import get_default_gd
from mf.get_mf_embedding import MF
from core.dataset import meta_data, dataset
from core.evalute import evalute, summary
from Helpers.util import cos_sim


# ***处理数据等最好不要放在recommend类中，并且该方法应设为recommend的子类？***
def Samanta(topK,if_pop=2,MF_mode='node2vec',pop_mode='',text_mode='HDP',LDA_topic_num=None):
    """
    :param Para:
    :param if_pop 如何使用pop  0 不使用；1，只做重排序；2总乘积做排序
    :param topK: 使用KNN表示新query的mf特征
    :param text_mode: 使用哪种特征提取方式  LDA  HDP
    :param pop_mode：pop值是否使用sigmoid规约到0-1区间
    :param pop_mode：MF_mode 为了省事，直接用node2vec得了
    :return:
    """

    api2pop=None
    if if_pop:
        api_co_vecs, api2pop = data_repository.get_md().get_api_co_vecs (pop_mode) # TODO

    root = os.path.join(data_repository.get_ds().data_root,'baselines')
    if not os.path.exists(root):
        os.makedirs(root)
    mashup_feature_path=os.path.join(root, 'mashup_{}.txt'.format(text_mode)) # ...
    api_feature_path = os.path.join(root, 'api_{}.txt'.format(text_mode))

    # 获取mashup_hdp_features,api_hdp_features
    if not os.path.exists(api_feature_path):
        gd=get_default_gd()
        _mashup_features,_api_features=gd.model_pcs(text_mode,LDA_topic_num)
        np.savetxt(mashup_feature_path,_mashup_features)
        np.savetxt(api_feature_path, _api_features)
    else:
        _mashup_features=np.loadtxt(mashup_feature_path)
        _api_features=np.loadtxt(api_feature_path)

    candidate_ids_list = []
    all_predict_results=[]

    test_data = data_repository.get_ds().test_data
    test_mashup_num = len(test_data.get('mashup'))
    mashup_emb_df = data_repository.get_ds().MF_obj.mashup_emb_df
    api_emb_df = data_repository.get_ds().MF_obj.api_emb_df

    for i in range(test_mashup_num):
        test_m_id=test_data.get('mashup')[i][0] # 每个mashup id
        candidate_ids = test_data.get('api')[i]
        candidate_ids_list.append(candidate_ids)

        # 用近邻mashup的latent factor加权表示自己
        mid2sim={}
        for train_m_id in mashup_emb_df.index.tolist():
            mid2sim[train_m_id]=cos_sim(_mashup_features[test_m_id],_mashup_features[train_m_id]) # TODO
        topK_ids,topK_sims=zip(*(sorted(mid2sim.items(), key=lambda x: x[1], reverse=True)[:topK]))
        topK_sims=np.array(topK_sims)/sum(topK_sims) # sim归一化
        cf_feature=np.zeros((data_repository.get_args().implict_feat_dim,))
        for z in range(len(topK_ids)):
            cf_feature += topK_sims[z] * mashup_emb_df['embedding'][topK_ids[z]]

        # 计算跟每个api的打分
        predict_results = []
        temp_predict_results=[] # 需要用pop进行重排序时的辅助
        api_zeros=np.zeros((data_repository.get_args().implict_feat_dim))
        api_ids = set(api_emb_df.index.tolist())
        for api_id in candidate_ids: # id
            api_i_feature= api_emb_df['embedding'][api_id]if api_id in api_ids else api_zeros  # 可能存在测试集中的api不在train中出现过的场景
            cf_score=np.sum(np.multiply(api_i_feature, cf_feature)) # mashup和api latent factor的内积
            sim_score=cos_sim(_mashup_features[test_m_id],_api_features[api_id]) # 特征的余弦相似度
            if if_pop==1:
                temp_predict_results.append((api_id,cf_score*sim_score))
            elif if_pop==0:
                predict_results.append(cf_score*sim_score)
            elif if_pop == 2:
                predict_results.append (cf_score * sim_score*api2pop[api_id])
        if if_pop==1:
            max_k_pairs = heapq.nlargest (topK, temp_predict_results, key=lambda x: x[1])  # 首先利用乘积排一次序
            max_k_candidates, _ = zip (*max_k_pairs)
            max_k_candidates=set(max_k_candidates)
            predict_results=[api2pop[api_id] if api_id in max_k_candidates else -1 for api_id in candidate_ids] # 重排序

        all_predict_results.append(predict_results)
    print('Samanta test,done!')

    evaluate_result = evalute(candidate_ids_list, all_predict_results, data_repository.get_ds().test_data.get('all_ground_api_ids'), data_repository.get_args().topKs)  # 评价
    _name='_pop_{}'.format(if_pop)
    _name+= data_repository.get_args().mf_mode
    csv_table_name = data_repository.get_ds().name + 'Samanta_model_{}'.format(topK)+_name + "\n"   # whole_model.name
    summary(evaluate_path, csv_table_name, evaluate_result, data_repository.get_args().topKs)  # 记录

    def divide(slt_apiNum):
        test_api_id_list_, predictions_, grounds_ = [], [], []
        for i in range(test_mashup_num):
            if len(data_repository.get_ds().slt_api_ids_instances[i]) == slt_apiNum:
                test_api_id_list_.append(candidate_ids_list[i])
                predictions_.append(all_predict_results[i])
                grounds_.append(data_repository.get_ds().test_data.get('all_ground_api_ids')[i])
        return test_api_id_list_, predictions_, grounds_
    if data_repository.get_args().data_mode == 'newScene':
        for slt_apiNum in range(3):
            test_api_id_list_, predictions_, grounds_ = divide(slt_apiNum+1)
            evaluate_result = evalute(test_api_id_list_, predictions_, grounds_, data_repository.get_args().topKs)
            summary(evaluate_path, str(slt_apiNum+1)+'_'+csv_table_name, evaluate_result, data_repository.get_args().topKs)  #



def hdp_pop(if_pop = True):
    # pop
    root = os.path.join(data_repository.get_ds().data_root,'baselines')
    if not os.path.exists(root):
        os.makedirs(root)
    mashup_hdp_path=os.path.join(root, 'mashup_HDP.txt') # ...
    api_hdp_path = os.path.join(root, 'api_HDP.txt')

    _mashup_hdp_features = np.loadtxt (mashup_hdp_path)
    _api_hdp_features = np.loadtxt (api_hdp_path)

    if if_pop:
        api_co_vecs, api2pop = meta_data.pd.get_api_co_vecs ()
    # 测试
    candidate_ids_list = []
    all_predict_results=[]
    for i in range(len(data_repository.get_ds().test_mashup_id_list)):
        test_mashup_id=data_repository.get_ds().test_mashup_id_list[i][0] # 每个mashup id
        candidate_ids = data_repository.get_ds().test_api_id_list[i]
        candidate_ids_list.append(candidate_ids)

        predict_results = []
        for api_id in candidate_ids: # id
            sim_score=cos_sim(_mashup_hdp_features[test_mashup_id],_api_hdp_features[api_id])
            if if_pop:
                sim_score *= api2pop[api_id]
            predict_results.append(sim_score)
        all_predict_results.append(predict_results)
    print('hdp_pop test,done!')

    evaluate_result = evalute(candidate_ids_list, all_predict_results, data_repository.get_ds().test_data.get('all_ground_api_ids'), data_repository.get_args().topKs)  # 评价
    name = 'hdp_pop' if if_pop else 'hdp'
    csv_table_name = data_repository.get_ds().name + name + "\n"   # whole_model.name
    summary(evaluate_path, csv_table_name, evaluate_result, data_repository.get_args().topKs)  # 记录


def TF_IDF(if_pop):
    """
    可以跟写到Samanta的类中，但太混乱，没必要
    :return:
    """
    gd = get_default_gd()
    api_co_vecs, api2pop = meta_data.pd.get_api_co_vecs()
    _mashup_IFIDF_features, _api_IFIDF_features = gd.model_pcs ('TF_IDF')

    candidate_ids_list = []
    all_predict_results=[]
    for i in range(len(data_repository.get_ds().test_mashup_id_list)):
        test_mashup_id=data_repository.get_ds().test_mashup_id_list[i][0] # 每个mashup id
        candidate_ids = data_repository.get_ds().test_api_id_list[i]
        candidate_ids_list.append(candidate_ids)

        predict_results = []
        for api_id in candidate_ids: # id
            sim_score=cos_sim(_mashup_IFIDF_features[test_mashup_id],_api_IFIDF_features[api_id])
            if if_pop:
                predict_results.append(sim_score*api2pop[api_id])
            else:
                predict_results.append(sim_score )
            predict_results.append(sim_score)
        all_predict_results.append(predict_results)
    print('TF_IDF test,done!')

    name = 'TFIDF_pop' if if_pop else 'TFIDF'
    evaluate_result = evalute(candidate_ids_list, all_predict_results, data_repository.get_ds().test_data.get('all_ground_api_ids'), data_repository.get_args().topKs)  # 评价
    csv_table_name = data_repository.get_ds().name + name + "\n"   # whole_model.name
    summary(evaluate_path, csv_table_name, evaluate_result, data_repository.get_args().topKs)  # 记录

def MF(train_datas,test_datas,mode = ''):
    all_predict_results=[] # 每个测试样例(多个api的)的评分
    for slt_num in range(1,data_repository.get_args().slt_item_num+1): # 不同个数的训练测试集
        test_mashup_id_list, test_api_id_list, grounds = test_datas[slt_num-1]
        # 增加处理和读取MF结果的接口
        UV_obj = MF(data_repository.get_ds().data_root, mode, train_datas[slt_num - 1], slt_num)
        m_id2index,a_id2index = UV_obj.m_id2index,UV_obj.a_id2index
        for i in range(len(test_mashup_id_list)):
            test_mashup_id=test_mashup_id_list[i][0] # 每个mashup id
            predict_results = []
            for test_api_id in test_api_id_list[i]: # id
                if test_mashup_id not in m_id2index or test_api_id not in a_id2index:
                    dot = 0
                else:
                    m_embedding = UV_obj.m_embeddings[m_id2index[test_mashup_id]]
                    a_embedding = UV_obj.a_embeddings[a_id2index[test_api_id]]
                    dot = np.dot(m_embedding,a_embedding)
                predict_results.append(dot)
            all_predict_results.append(predict_results)
        print('{}_{} test,done!'.format(mode,slt_num))

        evaluate_result = evalute(test_api_id_list, all_predict_results, data_repository.get_ds().test_data.get('all_ground_api_ids'), data_repository.get_args().topKs)  # 评价
        csv_table_name = data_repository.get_ds().name + mode + str(slt_num)+ "\n"   # whole_model.name
        summary(evaluate_path, csv_table_name, evaluate_result, data_repository.get_args().topKs)  # 记录

def pop():
    """
    :return:
    """
    api_co_vecs, api2pop = meta_data.pd.get_api_co_vecs ()
    candidate_ids_list = []
    all_predict_results=[]
    for i in range(len(data_repository.get_ds().test_mashup_id_list)):
        test_mashup_id=data_repository.get_ds().test_mashup_id_list[i][0] # 每个mashup id
        candidate_ids = data_repository.get_ds().test_api_id_list[i]
        candidate_ids_list.append(candidate_ids)

        predict_results = []
        for api_id in candidate_ids: # id
            predict_results.append(api2pop[api_id])
        all_predict_results.append(predict_results)
    print('pop test,done!')

    evaluate_result = evalute(candidate_ids_list, all_predict_results, data_repository.get_ds().test_data.get('all_ground_api_ids'), data_repository.get_args().topKs)  # 评价
    csv_table_name = data_repository.get_ds().name + 'pop' + "\n"   # whole_model.name
    summary(evaluate_path, csv_table_name, evaluate_result, data_repository.get_args().topKs)  # 记录

# """service package recommendation for mashup creation via mashup textual description mining"""


# “a novel approach for API recommendation in mashup development”
def binary_keyword(if_pop = False):
    # pop
    api_co_vecs, api2pop = meta_data.pd.get_api_co_vecs ()
    gd = get_default_gd()
    mashup_binary_matrix, api_binary_matrix, mashup_words_list, api_words_list = gd.get_binary_v ()


    # 测试WVSM(Weighted Vector Space Model)
    candidate_ids_list = []
    all_predict_results=[]
    for i in range(len(data_repository.get_ds().test_mashup_id_list)):
        test_mashup_id=data_repository.get_ds().test_mashup_id_list[i][0] # 每个mashup id
        candidate_ids = data_repository.get_ds().test_api_id_list[i]
        candidate_ids_list.append(candidate_ids)

        predict_results = []
        for api_id in candidate_ids: # id
            if if_pop:
                sim_score = cos_sim(mashup_binary_matrix[test_mashup_id], api_binary_matrix[api_id]) * api2pop[api_id]
            else:
                sim_score = cos_sim(mashup_binary_matrix[test_mashup_id], api_binary_matrix[api_id]) # 测试只使用特征向量的效果
            predict_results.append(sim_score)
        all_predict_results.append(predict_results)
    print('WVSM test,done!')

    evaluate_result = evalute(candidate_ids_list, all_predict_results, data_repository.get_ds().test_data.get('all_ground_api_ids'), data_repository.get_args().topKs)  # 评价
    name = 'WVSM_pop' if if_pop else 'WVSM'
    csv_table_name = data_repository.get_ds().name + name + "\n"   # whole_model.name
    summary(evaluate_path, csv_table_name, evaluate_result, data_repository.get_args().topKs)  # 记录

    """
    # 测试WJaccard(Weighted Jaccard)
    candidate_ids_list = []
    all_predict_results=[]
    for i in range(len(data_repository.get_ds().test_mashup_id_list)):
        test_mashup_id=data_repository.get_ds().test_mashup_id_list[i][0] # 每个mashup id
        candidate_ids = data_repository.get_ds().test_api_id_list[i]
        candidate_ids_list.append(candidate_ids)

        predict_results = []
        for api_id in candidate_ids: # id
            mashup_set=set(mashup_words_list[test_mashup_id])
            api_set = set (api_words_list[api_id])
            if if_pop:
                sim_score=1.0*len(mashup_set.intersection(api_set))/len(mashup_set.union(api_set))*api2pop[api_id]
            else:
                sim_score = 1.0 * len(mashup_set.intersection(api_set)) / len(mashup_set.union(api_set))
            predict_results.append(sim_score)
        all_predict_results.append(predict_results)
    print('WJaccard test,done!')

    evaluate_result = evalute(candidate_ids_list, all_predict_results, data_repository.get_ds().test_data.get('all_ground_api_ids'), data_repository.get_args().topKs)  # 评价
    name = 'WJaccard_pop' if if_pop else 'WJaccard'
    csv_table_name = data_repository.get_ds().name + name + "\n"   # whole_model.name
    summary(evaluate_path, csv_table_name, evaluate_result, data_repository.get_args().topKs)  # 记录
    """



if __name__=='__main__':
    # Samanta(topK, if_pop=1, MF_mode='pmf', pop_mode='')
    for mf in ['BPR', 'pmf', 'nmf', 'listRank']:  # 'pmf',
        for k in [10,20,30,40,50]: # ,100
            for if_pop in [1,2]:
                for pop_mode in ['']:# ,'sigmoid'
                    print('{},{},{},{}:'.format(k,if_pop,mf,pop_mode))
                    Samanta(k,if_pop=if_pop,MF_mode=mf,pop_mode=pop_mode)
    """"""
    # Samanta (50, if_pop=1, MF_mode='BPR') # 效果最好
    # TF_IDF()
    # binary_keyword ()  # 效果好的不敢相信。。相比之下我们的算法只提高了10%
    # pop()

    # hdp_pop () # Samanta使用pop*sim
