import os
import pickle
import sys

from core import data_repository
from recommend_models.recommend_Model import recommend_Model
from model_component.sequence import SequencePoolingLayer, AttentionSequencePoolingLayer, DNN
from model_component.simple_inception import inception_layer
from model_component.text_feature_extractors import HDP_feature_extracter_from_texts, \
    textCNN_feature_extracter_from_texts, LSTM_feature_extracter_from_texts
from model_component.utils import NoMask
from text_utils.gensim_data import get_default_gd
# from run_deepCTR.run_MISR_deepFM import transfer_testData2
from Helpers.util import save_2D_list
from text_utils.word_embedding import get_embedding_matrix

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Lambda, Concatenate, PReLU, BatchNormalization
from tensorflow.python.keras.layers import Dense, Input, Embedding, Multiply,Reshape
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import Adam


def slice(x, index):  # 三维的切片
    return x[:, index, :]


class CI_Model (recommend_Model):
    def __init__(self,args):
        super (CI_Model, self).__init__ (args)
        self.args = args
        self.pairwise_model = None
        self.predict_model = None
        self.optimizer = Adam(lr=self.args.CI_learning_rate)

        self.get_simple_name()
        self.set_paths()

        self.mashup_text_feature_extracter = None
        self.api_text_feature_extracter = None
        self.mashup_tag_feature_extracter = None
        self.api_tag_feature_extracter = None
        self.text_feature_extracter = None
        self.text_embedding_layer = None
        self.tag_embedding_layer = None

    def get_simple_name(self): # 记录关键的参数
        if not self.simple_name:
            if self.args.text_extracter_mode == 'inception':
                incep_mlp_setting = 'MLP_{}{}'.format('BN' if self.args.inception_MLP_BN else '',
                                                      'DP' if self.args.inception_MLP_dropout else '') if self.args.if_inception_MLP else ''
                incep_setting = 'POOL_{}_{}'.format(self.args.inception_pooling,incep_mlp_setting)
                text_extracter_setting = incep_setting
            elif self.args.text_extracter_mode == 'LSTM':
                text_extracter_setting = 'DIM_{}'.format(self.args.LSTM_dim)
            elif self.args.text_extracter_mode == 'textCNN':
                text_extracter_setting = 'CHA_{}'.format(str(self.args.textCNN_channels)).replace(',', '_')

            # mode:('CI', 'LR_PNCF')
            self.simple_name = '{}_{}_{}_{}_LR_{}_{}'.format(self.args.model_mode,self.args.text_extracter_mode,text_extracter_setting,
                                           self.args.simple_CI_slt_mode,self.args.CI_learning_rate,self.args.final_activation)
        return self.simple_name

    def set_paths(self):
        # 路径设置
        self.model_dir = os.path.join(data_repository.get_ds().data_root,self.get_simple_name()) # 模型路径
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_name_path = os.path.join(self.model_dir, 'model_name.dat')
        # self.CI_features_path = os.path.join(self.model_dir, 'CI_features.fea')
        # self.train_slt_apis_mid_features_path = os.path.join(self.model_dir, 'train_slt_apis_mid_features.csv')
        # self.test_slt_apis_mid_features_path = os.path.join(self.model_dir, 'test_slt_apis_mid_features.csv')
        self.ma_text_tag_feas_path = os.path.join(self.model_dir, 'mashup_api_text_tag_feas.dat')  # mashup和api的提取的文本特征

    def set_text_tag_enconding_layers(self):
        # 根据meta-data得到的文本和tag的编码表示，设置编码层
        all_mashup_num = data_repository.get_md().mashup_num
        mid2encoded_text = data_repository.get_md().mashup_df['padded_description'].tolist()
        mid2encoded_text = list(map(eval,mid2encoded_text))
        self.mashup_text_encoding_layer = Embedding(all_mashup_num+1, self.args.MAX_SEQUENCE_LENGTH,
                                             embeddings_initializer=Constant(mid2encoded_text),
                                             mask_zero=True, input_length=1,
                                             trainable=False, name='mashup_text_encoding_layer')

        all_api_num = data_repository.get_md().api_num
        aid2encoded_text = data_repository.get_md().api_df['padded_description'].tolist()
        aid2encoded_text = list(map(eval,aid2encoded_text))
        self.api_text_encoding_layer = Embedding(all_api_num+1, self.args.MAX_SEQUENCE_LENGTH,
                                             embeddings_initializer=Constant(aid2encoded_text),
                                             mask_zero=True, input_length=1,
                                             trainable=False, name='api_text_encoding_layer')

        mid2encoded_tags = data_repository.get_md().mashup_df['padded_categories'].tolist()
        mid2encoded_tags = list(map(eval,mid2encoded_tags))
        self.mashup_tag_encoding_layer = Embedding(all_mashup_num+1, self.args.MAX_TAGS_NUM,
                                            embeddings_initializer=Constant(mid2encoded_tags),
                                            mask_zero=True, input_length=1,
                                            trainable=False, name='mashup_tag_encoding_layer')

        aid2encoded_tags = data_repository.get_md().api_df['padded_categories'].tolist()
        aid2encoded_tags = list(map(eval,aid2encoded_tags))
        self.api_tag_encoding_layer = Embedding(all_api_num+1, self.args.MAX_TAGS_NUM,
                                            embeddings_initializer=Constant(aid2encoded_tags),
                                            mask_zero=True, input_length=1,
                                            trainable=False, name='api_tag_encoding_layer')


    def get_text_embedding_layer(self):
        """"
        得到定制的word embedding层,在feature_extracter_from_texts中使用
        """
        if self.text_embedding_layer is None:
            # 得到词典中每个词对应的embedding
            num_words = min(self.args.MAX_NUM_WORDS, len(data_repository.get_md().des_pd.word2index))+ 1  # 实际词典大小 +1  因为0代表0的填充向量
            temp = data_repository.get_md().des_pd.word2index


            self.text_embedding_matrix = get_embedding_matrix(data_repository.get_md().des_pd.word2index, self.args.embedding_name,
                                                              dimension=self.args.embedding_dim)
            print('built embedding matrix, done!')
            self.text_embedding_layer = Embedding(num_words,
                                                  self.args.embedding_dim,
                                                  embeddings_initializer=Constant(self.text_embedding_matrix),
                                                  embeddings_regularizer= regularizers.l2(self.args.embeddings_regularizer),
                                                  input_length=self.args.MAX_SEQUENCE_LENGTH,
                                                  mask_zero=True,
                                                  trainable=self.args.embedding_train, name = 'text_embedding_layer')  # mask_zero=True!!!
            print('built text embedding layer, done!')
        return self.text_embedding_layer

    def get_tag_embedding_layer(self):
        """"
        同text，处理tags,得到定制的word embedding层,在tag_feature_extractor中使用
        """
        if self.tag_embedding_layer is None:
            # 得到词典中每个词对应的embedding
            num_words = min(self.args.MAX_NUM_WORDS, len(data_repository.get_md().cate_pd.word2index))+ 1  # 实际词典大小 +1  因为0代表0的填充向量
            self.tag_embedding_matrix = get_embedding_matrix(data_repository.get_md().cate_pd.word2index, self.args.embedding_name,
                                                              dimension=self.args.embedding_dim)
            print('built tag embedding matrix, done!')
            self.tag_embedding_layer = Embedding(num_words,
                                                  self.args.embedding_dim,
                                                  embeddings_initializer=Constant(self.tag_embedding_matrix),
                                                  embeddings_regularizer=regularizers.l2(self.args.embeddings_regularizer),
                                                  input_length=self.args.MAX_TAGS_NUM,
                                                  mask_zero=True,
                                                  trainable=self.args.embedding_train,
                                                  name='tag_embedding_layer')
            print('built tag embedding layer, done!')
        return self.tag_embedding_layer

    def feature_extracter_from_texts(self,mashup_api=None):
        """
        对mashup，service的description均需要提取特征，右路的文本的整个特征提取过程
        公用的话应该封装成新的model！
        :param x:
        :return: 输出的是一个封装好的model，所以可以被mashup和api公用
        """
        if self.args.text_extracter_mode=='HDP' and mashup_api is not None:
            if self.gd is None:
                self.gd = get_default_gd(tag_times=1, mashup_only=False, strict_train=True)  # 用gensim处理文本,文本中不加tag
                self.gd.model_pcs(self.args.text_extracter_mode)  #

            if mashup_api == 'mashup':
                if self.mashup_text_feature_extracter is None: # 没求过
                    self.mashup_text_feature_extracter = HDP_feature_extracter_from_texts('mashup',self.gd.mashup_features)
                return self.mashup_text_feature_extracter
            elif mashup_api == 'api':
                if self.api_text_feature_extracter is None:
                    self.api_text_feature_extracter = HDP_feature_extracter_from_texts('api',self.gd.api_features)
                return self.api_text_feature_extracter

        if self.text_feature_extracter is None: # 没求过
            text_input = Input(shape=(self.args.MAX_SEQUENCE_LENGTH,), dtype='int32')
            text_embedding_layer = self.get_text_embedding_layer()  # 参数还需设为外部输入！
            text_embedded_sequences = text_embedding_layer(text_input)  # 转化为2D

            if self.args.text_extracter_mode in ('inception','textCNN'): # 2D转3D,第三维是channel
                # print(text_embedded_sequences.shape)
                text_embedded_sequences = Lambda(lambda x: tf.expand_dims(x, axis=3))(text_embedded_sequences)  # tf 和 keras的tensor 不同！！！
                print(text_embedded_sequences.shape)

            if self.args.text_extracter_mode=='inception':
                x = inception_layer(text_embedded_sequences, self.args.embedding_dim, self.args.inception_channels, self.args.inception_pooling)  # inception处理
                print('built inception layer, done!')
            elif self.args.text_extracter_mode=='textCNN':
                x = textCNN_feature_extracter_from_texts(text_embedded_sequences)
            elif self.args.text_extracter_mode=='LSTM':
                x = LSTM_feature_extracter_from_texts(text_embedded_sequences)
            else:
                raise TypeError('wrong extracter!')
            print('text feature after inception/textCNN/LSTM whole_model,',x) # 观察MLP转化前，模块输出的特征

            for FC_unit_num in self.args.inception_fc_unit_nums:
                x = Dense(FC_unit_num, kernel_regularizer=l2(self.args.l2_reg))(x)  # , activation='relu'
                if self.args.inception_MLP_BN:
                    x = BatchNormalization(scale=False)(x)
                x = PReLU()(x)  #
                if self.args.inception_MLP_dropout:
                    x = tf.keras.layers.Dropout(0.5)(x)
            self.text_feature_extracter=Model(text_input, x,name='text_feature_extracter')
        return self.text_feature_extracter

    def user_text_feature_extractor(self):
        if not self.mashup_text_feature_extracter:
            user_id_input = Input(shape=(1,), dtype='int32') # , name='mashup_id_input'
            mashup_text = self.mashup_text_encoding_layer(user_id_input)
            user_text_input = Lambda(lambda x: tf.cast(tf.squeeze(x, axis=1), 'int32'))(mashup_text)
            user_text_vec = self.feature_extracter_from_texts()(user_text_input) # (?,50)
            self.mashup_text_feature_extracter = Model(user_id_input, user_text_vec, name='user_text_feature_extracter')
        return self.mashup_text_feature_extracter

    # mashup和api的文本特征提取器，区别在于ID到文本编码的embedding矩阵不同，但又要公用相同的word_embedding层和inception特征提取器
    def item_text_feature_extractor(self):
        if not self.api_text_feature_extracter:
            item_id_input = Input(shape=(1,), dtype='int32') # , name='api_id_input'
            item_text_input = Lambda(lambda x: tf.cast(tf.squeeze(x, axis=1), 'int32'))(self.api_text_encoding_layer(item_id_input))
            item_text_vec = self.feature_extracter_from_texts()(item_text_input) # (?,50)
            self.api_text_feature_extracter = Model(item_id_input, item_text_vec, name='item_text_feature_extracter')
        return self.api_text_feature_extracter

    def user_tag_feature_extractor(self):
        if not self.mashup_tag_feature_extracter:
            user_id_input = Input(shape=(1,), dtype='int32') # , name='user_id_input'
            user_tag_input = Lambda(lambda x: tf.cast(tf.squeeze(x, axis=1), 'int32'))(self.mashup_tag_encoding_layer(user_id_input))
            user_tag_embedding = self.get_tag_embedding_layer()(user_tag_input)
            user_tag_vec = Lambda(lambda x: tf.squeeze(x, axis=1))(SequencePoolingLayer('mean', supports_masking=True)(user_tag_embedding))
            self.mashup_tag_feature_extracter = Model(user_id_input, user_tag_vec, name='user_tag_feature_extracter')
        return self.mashup_tag_feature_extracter

    def item_tag_feature_extractor(self):
        if not self.api_tag_feature_extracter:
            item_id_input = Input(shape=(1,), dtype='int32') # , name='api_id_input'
            item_tag_input = Lambda(lambda x: tf.cast(tf.squeeze(x, axis=1), 'int32'))(self.api_tag_encoding_layer(item_id_input))
            item_tag_embedding = self.get_tag_embedding_layer()(item_tag_input)
            item_tag_vec = Lambda(lambda x: tf.squeeze(x, axis=1))(SequencePoolingLayer('mean', supports_masking=True)(item_tag_embedding))
            self.api_tag_feature_extracter = Model(item_id_input, item_tag_vec, name='item_tag_feature_extracter')
        return self.api_tag_feature_extracter

    def get_model(self):
        if not self.model:
            mashup_id_input = Input(shape=(1,), dtype='int32', name='mashup_id_input')
            api_id_input = Input(shape=(1,), dtype='int32', name='api_id_input')
            inputs = [mashup_id_input, api_id_input]

            self.set_text_tag_enconding_layers()
            user_text_vec = self.user_text_feature_extractor()(mashup_id_input)
            item_text_vec = self.item_text_feature_extractor()(api_id_input)
            user_tag_vec = self.user_tag_feature_extractor()(mashup_id_input)
            item_tag_vec = self.item_tag_feature_extractor()(api_id_input)
            feature_list = [user_text_vec, item_text_vec, user_tag_vec, item_tag_vec]

            if self.args.model_mode == 'LR_PNCF':  # 旧场景，使用GMF形式的双塔模型
                x = Concatenate(name='user_concatenate')([user_text_vec, user_tag_vec])
                y = Concatenate(name='item_concatenate')([item_text_vec, item_tag_vec])
                output = Multiply()([x, y])
                predict_result = Dense(1, activation='sigmoid', use_bias=False, kernel_initializer='lecun_uniform',name="prediction")(output)  # 参数学习权重，非线性
                self.model = Model(inputs=inputs, outputs=[predict_result], name='predict_model')
                return self.model

            elif self.args.CI_handle_slt_apis_mode:
                # 需要处理已选择的服务
                mashup_slt_apis_input = Input(shape=(self.args.slt_item_num,), dtype='int32', name='slt_api_ids_input')
                mashup_slt_apis_input_3D = Reshape((self.args.slt_item_num,1))(mashup_slt_apis_input)
                # mashup_slt_apis_num_input = Input(shape=(1,), dtype='int32', name='mashup_slt_apis_num_input')
                inputs.append(mashup_slt_apis_input)
                mask = Lambda(lambda x: K.not_equal(x, 0))(mashup_slt_apis_input)  # (?, 3) !!!

                # 已选择的服务直接复用item_feature_extractor
                slt_text_vec_list,slt_tag_vec_list=[],[]
                for i in range(self.args.slt_item_num):
                    x = Lambda (slice, arguments={'index': i}) (mashup_slt_apis_input_3D) # (?,1,1)
                    x = Reshape((1,))(x)
                    temp_item_text_vec = self.item_text_feature_extractor()(x)
                    temp_item_tag_vec = self.item_tag_feature_extractor()(x)
                    slt_text_vec_list.append(temp_item_text_vec)
                    slt_tag_vec_list.append(temp_item_tag_vec)

                if self.args.CI_handle_slt_apis_mode in ('attention','average') :
                    # text和tag使用各自的attention block
                    slt_text_vec_list = [Reshape((1, self.args.embedding_dim))(key_2D) for key_2D in slt_text_vec_list]
                    slt_tag_vec_list = [Reshape((1, self.args.embedding_dim))(key_2D) for key_2D in slt_tag_vec_list]  # 增加了一维  eg:[None,50]->[None,1,50]
                    text_keys_embs = Concatenate(axis=1)(slt_text_vec_list)  # [?,3,50]
                    tag_keys_embs = Concatenate(axis=1)(slt_tag_vec_list)  # [?,3,50]

                    if self.args.CI_handle_slt_apis_mode == 'attention':
                        query_item_text_vec = Lambda(lambda x: tf.expand_dims(x, axis=1))(item_text_vec) # (?, 50)->(?, 1, 50)
                        query_item_tag_vec = Lambda(lambda x: tf.expand_dims(x, axis=1))(item_tag_vec)
                        # 压缩历史，得到向量
                        text_hist = AttentionSequencePoolingLayer(supports_masking=True)([query_item_text_vec, text_keys_embs],mask = mask)
                        tag_hist = AttentionSequencePoolingLayer(supports_masking=True)([query_item_tag_vec, tag_keys_embs],mask = mask)

                    else: # 'average'
                        text_hist = SequencePoolingLayer('mean', supports_masking=True)(text_keys_embs,mask = mask)
                        tag_hist = SequencePoolingLayer('mean', supports_masking=True)(tag_keys_embs,mask = mask)

                    text_hist = Lambda(lambda x: tf.squeeze(x, axis=1))(text_hist) # (?, 1, 50)->(?, 50)
                    tag_hist = Lambda(lambda x: tf.squeeze(x, axis=1))(tag_hist)

                elif self.args.CI_handle_slt_apis_mode == 'full_concate':
                    text_hist = Concatenate(axis=1)(slt_text_vec_list)  # [?,150]
                    tag_hist = Concatenate(axis=1)(slt_tag_vec_list)  # [?,150]
                else:
                    raise ValueError('wrong CI_handle_slt_apis_mode!')
                feature_list.extend([text_hist,tag_hist])

            else: # 包括新模型不处理已选择服务和旧模型
                pass
            feature_list = list(map(NoMask(), feature_list)) # DNN不支持mak，所以不能再传递mask
            all_features = Concatenate(name = 'all_content_concatenate')(feature_list)

            output = DNN(self.args.content_fc_unit_nums[:-1])(all_features)
            output = Dense (self.args.content_fc_unit_nums[-1], activation='relu', kernel_regularizer = l2(self.args.l2_reg), name='text_tag_feature_extracter') (output)

            # 输出层
            if self.args.final_activation=='softmax':
                predict_result = Dense(2, activation='softmax', name="prediction")(output)
            elif self.args.final_activation=='sigmoid':
                predict_result = Dense (1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction") (output)

            print('predict_result',predict_result)
            self.model = Model (inputs=inputs,outputs=[predict_result],name='predict_model')
            for layer in self.model.layers:
                print(layer.name)
            print('built CI whole_model, done!')
        return self.model

    def show_text_tag_features(self, train_data, show_num=10):
        """
        检查生成的mashup和api的text和tag的特征是否正常
        """
        if not self.args.CI_handle_slt_apis_mode:
            m_ids,a_ids = train_data[:-1]
            instances_tuple = self.get_instances(m_ids[:show_num],a_ids[:show_num])
        else:
            m_ids,a_ids,slt_a_ids = train_data[:-1]
            instances_tuple = self.get_instances(m_ids[:show_num],a_ids[:show_num],slt_a_ids[:show_num])

        text_tag_middle_model = Model(inputs=[*self.model.inputs],
                                      outputs=[*self.model.get_layer('all_content_concatenate').input[:4]])
        mashup_text_features,apis_text_features, mashup_tag_features,apis_tag_features = text_tag_middle_model.predict([*instances_tuple], verbose=0)

        mashup_text_features_path = os.path.join(self.model_dir,'mashup_text_features.dat')
        apis_text_features_path = os.path.join(self.model_dir,'apis_text_features.dat')
        mashup_tag_features_path = os.path.join(self.model_dir,'mashup_tag_features.dat')
        apis_tag_features_path = os.path.join(self.model_dir,'apis_tag_features.dat')

        save_2D_list(mashup_text_features_path, mashup_text_features, 'a+')
        save_2D_list(apis_text_features_path, apis_text_features, 'a+')
        save_2D_list(mashup_tag_features_path, mashup_tag_features, 'a+')
        save_2D_list(apis_tag_features_path, apis_tag_features, 'a+')

    def get_pairwise_model(self):
        if self.pairwise_model is None:
            if self.args.pairwise and self.model is not None:  # 如果使用pairwise型的目标函数
                mashup_id_input = Input (shape=(1,), dtype='int32', name='mashup_id_input')
                api_id_input = Input (shape=(1,), dtype='int32', name='api_id_input')
                neg_api_id_input = Input(shape=(1,), dtype='int32', name='neg_api_id_input')
                mashup_slt_apis_input = Input(shape=(self.args.slt_item_num,), dtype='int32', name='slt_api_ids_input')
                if self.args.CI_handle_slt_apis_mode:
                    pos_ratings = self.model([mashup_id_input, api_id_input, mashup_slt_apis_input])
                    neg_ratings = self.model([mashup_id_input, neg_api_id_input, mashup_slt_apis_input])  # 再加一个负例api id
                else:
                    pos_ratings = self.model([mashup_id_input, api_id_input])
                    neg_ratings = self.model([mashup_id_input, neg_api_id_input])
                # pairwise hinge loss; 可以考虑使用BPR:exp形式的loss
                loss = Lambda(lambda x: K.relu(self.args.margin + x[0] - x[1]),name='sub_result')([neg_ratings, pos_ratings])

                # 注意输入格式！
                if self.args.CI_handle_slt_apis_mode:
                    self.pairwise_model = Model(inputs=[mashup_id_input, api_id_input, mashup_slt_apis_input, neg_api_id_input],outputs=loss)
                else:
                    self.pairwise_model = Model(inputs=[mashup_id_input, api_id_input, neg_api_id_input], outputs=loss)

                for layer in self.pairwise_model.layers:
                    print(layer.name)

                # # 复用的是同一个对象！
                # print(self.pairwise_model.get_layer('predict_model'),id(self.pairwise_model.get_layer('predict_model')))
                # print(self.whole_model,id(self.whole_model))
        return self.pairwise_model

    def get_instances(self, data, pairwise_train_phase_flag=False):
        """
        生成该模型需要的样本
        slt_api_ids_instances是每个样本中，已经选择的api的id序列  变长二维序列
        train和test样例都可用  但是针对一维列表形式，所以测试时先需拆分（text的数据是二维列表）！！！
        :param args:
        :return:
        """
        instances = {'mashup_id_input':np.array(data.get('mashup')),
                     'api_id_input':np.array(data.get('api'))
                     }
        # if self.args.need_slt_apis and slt_api_ids_instances: # 是否加入slt_api_ids_instances
        slt_api_ids_instances = data.get('slt_apis')
        if self.args.CI_handle_slt_apis_mode: # 根据模型变化调整决定，同时输入的数据本身也是对应的
            # 节省内存版, 不够slt_item_num的要padding
            instance_num = len(slt_api_ids_instances)
            padded_slt_api_instances = np.zeros((instance_num, self.args.slt_item_num),dtype='int32')
            for i in range(instance_num):
                a_slt_api_ids = slt_api_ids_instances[i]
                padded_slt_api_instances[i][:len(a_slt_api_ids)] = a_slt_api_ids
            instances['slt_api_ids_input'] = padded_slt_api_instances

        #  pairwise test时，不需要neg_api_id_instances

        if self.args.pairwise and pairwise_train_phase_flag:
            instances['neg_api_id_input'] = np.array(data.get('neg_api'))

        return instances

    def get_mashup_api_features(self):
        """
        得到每个mashup和api经过特征提取器或者平均池化得到的特征，可以直接用id索引，供构造instance的文本部分使用
        :param text_tag_recommend_model:
        :param mashup_num:
        :param api_num:
        :return:
        """
        if os.path.exists(self.ma_text_tag_feas_path):
            with open(self.ma_text_tag_feas_path,'rb') as f1:
                mashup_texts_features, mashup_tag_features, api_texts_features, api_tag_features = pickle.load(f1)
        else:
            # 前四个分别是 user_text_vec, item_text_vec, user_tag_vec, item_tag_vec
            text_tag_middle_model = Model(inputs=[*self.model.inputs[:2]],
                                          outputs=[self.model.get_layer('all_content_concatenate').input[0],
                                                   self.model.get_layer('all_content_concatenate').input[1],
                                                   self.model.get_layer('all_content_concatenate').input[2],
                                                   self.model.get_layer('all_content_concatenate').input[3]])

            feature_mashup_ids = data_repository.get_md().mashup_df.index.tolist()
            feature_instances_tuple = self.get_instances(feature_mashup_ids, [0] * len(feature_mashup_ids))
            mashup_texts_features,_1, mashup_tag_features,_2 = text_tag_middle_model.predict ([*feature_instances_tuple], verbose=0)

            feature_api_ids = data_repository.get_md().api_df.index.tolist()
            feature_instances_tuple = self.get_instances([0] * len(feature_api_ids),feature_api_ids)
            _1,api_texts_features, _2,api_tag_features = text_tag_middle_model.predict ([*feature_instances_tuple], verbose=0)

            with open(self.ma_text_tag_feas_path, 'wb') as f2:
                pickle.dump((mashup_texts_features, mashup_tag_features, api_texts_features, api_tag_features),f2)
        return mashup_texts_features,mashup_tag_features,api_texts_features,api_tag_features


