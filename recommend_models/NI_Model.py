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

class NI_Model(recommend_Model):
    def __init__(self, args):
        super(NI_Model, self).__init__(args)

        self.args = args
        self.optimizer = Adam(self.args.NI_learning_rate)
        self.model = None

        self.get_simple_name()
        self.set_paths()

    def get_simple_name(self):
        if not self.simple_name:
            if self.args.if_correlation:
                self.simple_name= 'COR_'.format(self.args.cor_fc_unit_nums).replace(',', '_')
            else:
                sim_name = '{}_{}_{}'.format(self.args.NI_sim_mode,self.args.path_topK_mode,self.args.topK)
                if self.args.if_explict:
                    subfix='EXP_{}'.format(self.args.exp_fc_unit_nums).replace(',', '_')
                elif self.args.if_implict:
                    subfix='IMP_{}_DIM{}_{}_{}_LR_{}_{}'.format(self.args.mf_mode,self.args.implict_feat_dim,self.args.imp_fc_unit_nums,
                        self.args.simple_NI_slt_mode, self.args.NI_learning_rate, self.args.final_activation).replace(',', '_')
                self.simple_name = sim_name+''+subfix
            # self.simple_name += '_reductData' # !!!
        return self.simple_name

    def set_paths(self):
        self.model_dir = os.path.join(data_repository.get_ds().data_root,self.get_simple_name()) # 模型路径
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_name_path = os.path.join(self.model_dir, 'model_name.dat')
        # self.NI_features_path = os.path.join(self.model_dir, 'NI_features.fea')
        # self.train_slt_apis_mid_features_path = os.path.join(self.model_dir,'train_slt_apis_mid_features.csv')
        # self.test_slt_apis_mid_features_path = os.path.join(self.model_dir, 'test_slt_apis_mid_features.csv')

    def set_mashup_api_features(self, recommend_model):
        """
        TODO
        设置mashup和api的text和tag特征，用于计算相似度，进而计算mashup的NI表示;
        在get_model()和get_instances()之前设置
        :param recommend_model: 利用CI模型获得所有特征向量
        :return:
        """
        self.mashup_texts_features, self.mashup_tag_features, self.api_texts_features, self.api_tag_features = \
            recommend_model.get_mashup_api_features(data_repository.get_md().mashup_num, data_repository.get_md().api_num)
        # api 需要增加一个全为0的，放在最后，id为api_num，用来对slt_apis填充
        self.api_tag_features = np.vstack((self.api_tag_features, np.zeros((1, self.word_embedding_dim))))
        self.api_texts_features = np.vstack((self.api_texts_features, np.zeros((1, self.inception_fc_unit_nums[-1]))))
        self.features = (self.mashup_texts_features, self.mashup_tag_features, self.api_texts_features, self.api_tag_features)
        self.CI_path = recommend_model.model_dir

    def prepare(self,sim_model, train_data, test_data):
        """
        计算新mashup和api的embedding表示
        :param sim_model: 基于哪种相似度模型计算mashup相似度
        :param train_data:
        :param test_data:
        :return:
        """
        # 计算新mashup的表示
        self.new_m_fea_obj = get_NI_new_feature(self.args)
        self.new_m_fea_obj.process(sim_model, train_data, test_data)

    def set_embedding_matrixs(self):
        # id->embedding
        self.i_factors_matrix = np.zeros((data_repository.get_md().api_num + 1, self.args.implict_feat_dim))
        api_emb_df = data_repository.get_ds().MF_obj.api_emb_df
        for row in zip(api_emb_df.index.tolist(),api_emb_df.embedding.tolist()):
            id,embedding = row[0],row[1]
            if isinstance(embedding,str):
                embedding = eval(embedding)
            self.i_factors_matrix[id] = embedding

    def set_embedding_layers(self):
        self.api_implict_emb_layer = Embedding(data_repository.get_md().api_num + 1,
                                               self.args.implict_feat_dim,
                                               embeddings_initializer=Constant(self.i_factors_matrix),
                                               mask_zero=False,
                                               trainable=False,
                                               name='api_implict_embedding_layer')

    def get_model(self):
        if not self.model:
            # 设置api embedding矩阵
            self.set_embedding_matrixs()
            self.set_embedding_layers()

            mashup_fea_input = Input(shape=(self.args.implict_feat_dim,), dtype='float32', name='NI_mashup_fea_input')  # (None,25)
            api_id_input = Input(shape=(1,), dtype='int32', name='NI_api_id_input')
            inputs = [mashup_fea_input, api_id_input]

            api_implict_embs = self.api_implict_emb_layer(api_id_input)  # (None,1,25)
            api_implict_embs_2D = Lambda(lambda x: tf.squeeze(x, axis=1))(api_implict_embs) # (None,25)
            feature_list = [mashup_fea_input,api_implict_embs_2D]

            if self.args.NI_handle_slt_apis_mode:
                mashup_slt_apis_input = Input(shape=(self.args.slt_item_num,), dtype='int32', name='slt_apis_input')
                inputs.append(mashup_slt_apis_input)
                keys_slt_api_implict_embs = self.api_implict_emb_layer(mashup_slt_apis_input)  # (None,3,25)

                if self.args.NI_handle_slt_apis_mode in ('attention','average') :
                    mask = Lambda(lambda x: K.not_equal(x, 0))(mashup_slt_apis_input)  # (?, 3) !!!
                    if self.args.NI_handle_slt_apis_mode == 'attention':
                        slt_api_implict_embs_hist = AttentionSequencePoolingLayer(supports_masking=True)([api_implict_embs, keys_slt_api_implict_embs],mask=mask)
                    else:  # 'average'
                        slt_api_implict_embs_hist = SequencePoolingLayer('mean', supports_masking=True)(keys_slt_api_implict_embs, mask=mask)
                    slt_api_implict_embs_hist = Lambda(lambda x: tf.squeeze(x, axis=1))(slt_api_implict_embs_hist)  # (?, 1, 25)->(?, 25)
                elif self.args.NI_handle_slt_apis_mode == 'full_concate':
                    slt_api_implict_embs_hist = Reshape((self.args.slt_item_num*self.args.implict_feat_dim,))(keys_slt_api_implict_embs)  # (?,75)
                else:
                    raise ValueError('wrong NI_handle_slt_apis_mode!')
                feature_list.append(slt_api_implict_embs_hist)

            feature_list = list(map(NoMask(), feature_list))  # DNN不支持mak，所以不能再传递mask
            all_features = Concatenate(name='all_emb_concatenate')(feature_list)

            output = DNN(self.args.imp_fc_unit_nums[:-1])(all_features)
            output = Dense(self.args.imp_fc_unit_nums[-1], activation='relu', kernel_regularizer=l2(self.args.l2_reg),name='implict_dense_{}'.format(len(self.args.imp_fc_unit_nums)))(output)

            # 输出层
            if self.args.final_activation == 'softmax':
                predict_result = Dense(2, activation='softmax', name="prediction")(output)
            elif self.args.final_activation == 'sigmoid':
                predict_result = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(output)

            self.model = Model(inputs=inputs, outputs=[predict_result], name='predict_model')

            for layer in self.model.layers:
                print(layer.name)
            if not os.path.exists(self.model_name_path):
                with open(self.model_name_path, 'w+') as f:
                    f.write(self.get_name())
        return self.model

    def get_instances(self, data,pairwise_train_phase_flag=False):
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
                [self.new_m_fea_obj.m2NI_feas[(mashup_id_list[i], tuple(mashup_slt_apis_list[i]))]
                 for i in range(len(mashup_id_list))], dtype='float32')
        elif self.args.NI_sim_mode == 'PasRec_2path' or self.args.NI_sim_mode == 'IsRec_best':
            mashup_fea_array = np.array([self.new_m_fea_obj.m2NI_feas[m_id] for m_id in mashup_id_list])
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