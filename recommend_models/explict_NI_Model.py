import os
import sys

from core import data_repository
from model_component.sequence import AttentionSequencePoolingLayer, SequencePoolingLayer,DNN
from model_component.utils import NoMask
from Helpers.get_NI_new_feature import get_NI_new_feature
from recommend_models.recommend_Model import recommend_Model

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import Dense, Input, Concatenate, Embedding, Lambda, Reshape
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import Constant


# 针对新场景，在内容交互的基础上搭建新的完整模型:
# 可以完全依赖online_node2vec得到新mashup的隐式表示
# NI部分只使用tag feature+ cosine作为mashup相似度，暂不考虑已选择服务，换句话说，为每个mashup固定近邻和node2vec空间的特征表示

class explict_NI_Model(recommend_Model):
    def __init__(self, args):
        super(explict_NI_Model, self).__init__(args)

        self.args = args
        self.optimizer = Adam(self.args.NI_learning_rate)
        self.model = None

        self.get_simple_name()
        self.set_paths()

    def get_simple_name(self):
        if not self.simple_name:
            sim_name = '{}_{}_{}'.format(self.args.NI_sim_mode, self.args.path_topK_mode, self.args.topK)
            if self.args.if_correlation:
                subfix = 'COR_'.format(self.args.cor_fc_unit_nums).replace(',', '_')
            elif self.args.if_explict:
                subfix='EXP_{}'.format(self.args.exp_fc_unit_nums).replace(',', '_')
            self.simple_name = sim_name+'_'+subfix
        return self.simple_name

    def set_paths(self):
        self.model_dir = os.path.join(data_repository.get_ds().data_root,self.get_simple_name()) # 模型路径
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_name_path = os.path.join(self.model_dir, 'model_name.dat')
        # self.NI_features_path = os.path.join(self.model_dir, 'NI_features.fea')
        # self.train_slt_apis_mid_features_path = os.path.join(self.model_dir,'train_slt_apis_mid_features.csv')
        # self.test_slt_apis_mid_features_path = os.path.join(self.model_dir, 'test_slt_apis_mid_features.csv')

    def prepare(self,sim_model, train_data, test_data,new_m_fea_obj=None):
        """
        计算新mashup和api的embedding表示
        :param sim_model: 基于哪种相似度模型计算mashup相似度
        :param train_data:
        :param test_data:
        :return:
        """
        if new_m_fea_obj is None:
            # 计算新mashup的表示
            self.new_m_fea_obj = get_NI_new_feature(self.args)
            self.new_m_fea_obj.process(sim_model, train_data, test_data)
        else:
            self.new_m_fea_obj = new_m_fea_obj

        # 根据self.new_m_fea_obj.mid2neighors和其他信息构造...

    def get_model(self):
        if not self.model:
            # 输入形状根据设置有不同
            input = Input(shape=(self.args.implict_feat_dim,), dtype='float32', name='NI_mashup_fea_input')  # (None,25)

            output = DNN(self.args.imp_fc_unit_nums[:-1])(input)
            output = Dense(self.args.imp_fc_unit_nums[-1], activation='relu', kernel_regularizer=l2(self.args.l2_reg),name='implict_dense_{}'.format(len(self.args.imp_fc_unit_nums)))(output)

            # 输出层
            if self.args.final_activation == 'softmax':
                predict_result = Dense(2, activation='softmax', name="prediction")(output)
            elif self.args.final_activation == 'sigmoid':
                predict_result = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(output)

            self.model = Model(inputs=[input], outputs=[predict_result], name='predict_model')

            for layer in self.model.layers:
                print(layer.name)
            if not os.path.exists(self.model_name_path):
                with open(self.model_name_path, 'w+') as f:
                    f.write(self.get_name())
        return self.model

    def get_instances(self, data,pairwise_train_phase_flag=False):
        # 根据self.new_m_fea_obj.mid2neighors和其他信息构造...

        mashup_id_list = data.get('mashup')
        api_id_list = data.get('api')
        mashup_slt_apis_list = data.get('slt_apis')

        # if self.args.if_explict:
        #     co_vecs = np.array([[1.0 if api_id_list[i] in data_repository.get_ds().train_mashup_api_dict[neighbor_id] else 0.0
        #                          for neighbor_id in self.new_m_fea_obj.m2neighors[mashup_id_list[i]]]
        #                          for i in range(len(mashup_id_list))])
        #     return co_vecs

        if self.args.NI_sim_mode == 'PasRec' or self.args.NI_sim_mode == 'IsRec':
            mashup_fea_array = np.array(
                [self.new_m_fea_obj.mid_sltAids_2NI_feas[(mashup_id_list[i], tuple(mashup_slt_apis_list[i]))]
                 for i in range(len(mashup_id_list))], dtype='float32')
        elif self.args.NI_sim_mode == 'PasRec_2path' or self.args.NI_sim_mode == 'IsRec_best':
            mashup_fea_array = np.array([self.new_m_fea_obj.mid_sltAids_2NI_feas[m_id] for m_id in mashup_id_list])
        else:
            raise TypeError('wrong NI_sim_mode!')

        instances = {'NI_mashup_fea_input':mashup_fea_array,
                     'NI_api_id_input':np.array(api_id_list)
                     }
        if self.args.NI_handle_slt_apis_mode:
            # pad using 0 (placeholder when encoding)
            mashup_slt_apis_array = np.zeros((len(mashup_id_list), self.args.slt_item_num),dtype='int32')
            for i in range(len(mashup_slt_apis_list)):
                for j in range(len(mashup_slt_apis_list[i])):
                    mashup_slt_apis_array[i][j] = mashup_slt_apis_list[i][j]
            instances['slt_apis_input'] = mashup_slt_apis_array

        # TODO pairwise
        return instances