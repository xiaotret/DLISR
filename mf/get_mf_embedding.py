import os
import pickle
import sys
from collections import OrderedDict
import numpy as np
import pandas as pd

from Helpers.util import get_id2index, save_id2index, write_mashup_api_pair, get_mf_embedding_df
from mf.Node2Vec import call_node2vec, args, get_newNode_vec


class MF():
    def __init__(self, data_path, mode, slt_num=0):
        """
        根据训练集，调用其他MF/GE的库，得到mashup/api的embedding
        :param data_path: 哪种数据
        :param mode: 使用哪种矩阵分解方法
        """
        if slt_num == 0:
            self.MF_root = os.path.join(data_path, 'U_V', mode)
        else:
            self.MF_root = os.path.join(data_path, 'U_V', str(slt_num) + '_' + mode)  # 1_BPR
        if not os.path.exists(self.MF_root):
            os.makedirs(self.MF_root)

        self.mashup_emb_df, self.api_emb_df = None, None

    def get_mf_embedding(self,train_mashup_api_list,mode):
        """
        :param train_mashup_api_list: 训练集
        :param mode:
        :return: df形式，id是其索引, embedding列的列名是embedding
        """
        if self.mashup_emb_df is None or self.api_emb_df is None:
            if mode == 'node2vec':
                self.mashup_emb_df, self.api_emb_df = get_UV_from_Node2vec(self.MF_root, train_mashup_api_list)
            elif mode == 'BiNE':
                rating_train_path = os.path.join(self.MF_root, 'rating_train.dat')
                if not os.path.exists(rating_train_path):
                    prepare_data_for_BiNE(train_mashup_api_list, rating_train_path)
                    print('you ought to run BiNE first!!!')
                    sys.exit()
                    # 有时间将BiNE的代码整合到该工程中
                self.mashup_emb_df = get_BiNE_UI_embeddings(os.path.join(self.MF_root, 'vectors_u.dat'))
                self.api_emb_df = get_BiNE_UI_embeddings(os.path.join(self.MF_root, 'vectors_v.dat'))
            else:
                self.mashup_emb_df = get_UV_from_librec(self.MF_root, "mashup")
                self.api_emb_df = get_UV_from_librec(self.MF_root, "api")
        return self.mashup_emb_df ,self.api_emb_df


def get_UV_from_librec(MF_path, user_or_item):
    """
    读取从librec得到的结果，结果是df(index是mashup/api的id)
    :param MF_path:
    :param user_or_item:
    :param ordered_ids: 一般是按照mashup，api的id从小到大排列的
    :return:
    """
    if user_or_item == "mashup":
        id2index_path = os.path.join(MF_path, "userIdToIndex.csv")
        matrix_path = os.path.join(MF_path, "U.txt")
    elif user_or_item == "api":
        id2index_path = os.path.join(MF_path, "itemIdToIndex.csv")
        matrix_path = os.path.join(MF_path, "V.txt")
    else:
        raise TypeError('user_or_item must be mashup or api!')

    if not os.path.exists(matrix_path):
        raise IOError('MUST RUN librec in java first!')

    matrix = np.loadtxt(matrix_path).tolist()
    emb_df = pd.DataFrame(data={'embedding': matrix, 'index': list(range(len(matrix)))})  # index从0开始；必须用字典的形式传递list
    df = get_mf_embedding_df(emb_df, id2index_path)
    return df


def prepare_data_for_Node2vec(a_args, train_mashup_api_list):
    """
    :param train_mashup_api_list: # 需传入内部索引？？？外部
    :return:
    """
    m_ids, a_ids = zip(*train_mashup_api_list)
    m_ids = np.unique(m_ids)
    a_ids = np.unique(a_ids)
    m_num = len(m_ids)

    # 对mashup和api的id统一编码：index
    m_id2index = {m_ids[index]: index + 1 for index in range(m_num)}
    save_id2index(m_id2index, a_args.m_id_map_path)
    a_id2index = {a_ids[index]: m_num + index + 1 for index in range(len(a_ids))}
    save_id2index(a_id2index, a_args.a_id_map_path)

    pair = []
    for m_id, a_id in train_mashup_api_list:
        pair.append((m_id2index[m_id], a_id2index[a_id]))  # 写入编码后的内部索引
    write_mashup_api_pair(pair, a_args.input, 'list')
    print('prepare_data_for_Node2vec,done!')


def get_UV_from_Node2vec(node2vec_res_root, train_mashup_api_list):
    """
    传入U-I,返回mashup和api的embedding矩阵,按照id大小排列
    :param node2vec_res_root:结果root
    :param train_mashup_api_list:
    :return:
    """
    a_args = args(node2vec_res_root)
    if not os.path.exists(a_args.m_embedding):
        prepare_data_for_Node2vec(a_args, train_mashup_api_list)
        call_node2vec(a_args)
        index2embedding = OrderedDict()
        with open(a_args.output, 'r') as f:
            f.readline()  # 跳过第一行的信息
            line = f.readline()
            while line:
                l = line.split(' ')
                index = int(l[0])
                embedding = [float(value) for value in l[1:]]
                index2embedding[index] = embedding
                line = f.readline()
        emb_df = pd.DataFrame(data={'index': list(index2embedding.keys()), 'embedding': list(index2embedding.values())})
        mashup_emb_df = get_mf_embedding_df(emb_df, a_args.m_id_map_path)  # 可以从结果df中，根据id获取embedding
        api_emb_df = get_mf_embedding_df(emb_df, a_args.a_id_map_path)

        mashup_emb_df.to_csv(a_args.m_embedding)
        api_emb_df.to_csv(a_args.a_embedding)
    else:
        mashup_emb_df = pd.read_csv(a_args.m_embedding).set_index('id')
        api_emb_df = pd.read_csv(a_args.a_embedding).set_index('id')
    return mashup_emb_df, api_emb_df


def prepare_data_for_BiNE(train_mashup_api_list, result_path):
    with open(result_path, 'w') as f:
        for m_id, a_id in train_mashup_api_list:
            f.write("u{}\ti{}\t{}\n".format(m_id, a_id, 1))


def get_BiNE_UI_embeddings(result_path):
    id2embeddings = OrderedDict()
    with open(result_path, 'r') as f:
        line = f.readline()
        while line:
            a_line = line.strip().split(" ")
            id = int(a_line[0][1:])
            embedding = [float(value) for value in a_line[1:]]
            id2embeddings[id] = embedding
            line = f.readline()
    df = pd.DataFrame(data={'id': list(id2embeddings.keys()), 'embedding': list(id2embeddings.values())})
    df = df.set_index('id')
    return df
