# -*- coding:utf-8 -*-
from core import data_repository
from Helpers.cpt_DHSR_Sim import get_sims_dict

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense, Input, Embedding, concatenate, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam


class recommend_Model(object):
    """
    共同基类
    """
    def __init__(self,args):
        self.model = None
        self.args = args
        self.simple_name,self.name =None, None

    def get_simple_name(self):
        if not self.simple_name:
            self.simple_name = ''
        return self.simple_name

    def get_name(self):
        """
        用在记录结果部分，记录数据信息+模型信息
        :return:
        """
        if not self.name:
            self.name = data_repository.get_md().name + '_' + data_repository.get_ds().name + '_' + self.simple_name
        return self.name

    def set_paths(self):
        self.model_dir = data_repository.get_ds().model_path.format(self.get_simple_name())  # 模型路径

    def get_model(self):
        """
        **TO OVERIDE**
        :return:  a whole_model
        """
        pass

    def get_instances(self):
        """
        在模型文件中把ID形式的样本转化为模型需要的输入
        **TO OVERIDE**
        """
        pass

    def save_sth(self):
        pass


class DHSR_model(recommend_Model):
    def __init__(self,args,slt_num=0):
        """
        :param slt_num: 是否约减数据...
        """
        super(DHSR_model, self).__init__(args)
        self.slt_num = slt_num
        self.sims_dict = get_sims_dict(False,True) # 相似度对象，可改参数？
        self.optimizer = Adam(lr=self.args.DHSR_lr)
        self.get_simple_name()
        self.set_paths()

    def get_simple_name(self):
        if not self.simple_name:
            self.simple_name = 'DHSR{}_MF_MLP{}_DIM{}_SimSize{}_{}'.format(
                '_{}'.format(self.slt_num) if self.slt_num > 0 else '', self.args.mf_fc_unit_nums,
                self.args.mf_embedding_dim, self.args.sim_feature_size, self.args.final_MLP_layers).replace(',', ' ')
        return self.simple_name

    def get_model(self):
        # Input Layer
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')
        text_input = Input(shape=(self.args.sim_feature_size,), dtype='float32', name='text_input')

        # Embedding layer
        MF_Embedding_User = Embedding(input_dim=self.num_users, output_dim=self.args.mf_embedding_dim, name='mf_embedding_user',
                                      embeddings_initializer=initializers.VarianceScaling(scale=0.01,distribution='normal'),
                                      embeddings_regularizer=l2(0.01), input_length=1)
        MF_Embedding_Item = Embedding(input_dim=self.num_items, output_dim=self.args.mf_embedding_dim, name='mf_embedding_item',
                                      embeddings_initializer=initializers.VarianceScaling(scale=0.01,distribution='normal'),
                                      embeddings_regularizer=l2(0.01), input_length=1)
        # MF part
        mf_user_latent = tf.keras.layers.Flatten()(MF_Embedding_User(user_input))
        mf_item_latent = tf.keras.layers.Flatten()(MF_Embedding_Item(item_input))  # why Flatten？
        mf_vector = concatenate([mf_user_latent, mf_item_latent])  # element-wise multiply    ???

        for idx in range(len(self.args.mf_fc_unit_nums)):   # 学习非线性关系
            layer = Dense(self.args.mf_fc_unit_nums[idx],  activation='relu', name="layer%d" % idx)
            mf_vector = layer(mf_vector)

        # Text part
        # text_input = Dense(10, activation='relu', kernel_regularizer=l2(0.01))(text_input)  #   sim? 需要再使用MLP处理下？

        # Concatenate MF and TEXT parts
        predict_vector = concatenate([mf_vector, text_input])

        for idx in range(len(self.args.final_MLP_layers)):   # 整合后再加上MLP？
            layer = Dense(self.args.final_MLP_layers[idx], activation='relu')# name="layer%d"  % idx
            predict_vector = layer(predict_vector)

        predict_vector = tf.keras.layers.Dropout(0.5)(predict_vector)    # 使用dropout?

        if self.args.final_activation == 'softmax':
            predict_vector = Dense(2, activation='softmax', name="prediction")(predict_vector)
        elif self.args.final_activation == 'sigmoid':
            predict_vector = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(predict_vector)

        # # Final prediction layer
        # predict_vector = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(predict_vector)

        model = Model(inputs=[user_input, item_input, text_input],outputs=predict_vector)
        return model

    def get_instances(self,mashup_id_instances, api_id_instances,if_Train=False, test_phase_flag=False):
        sims=[]
        for i in range(len(mashup_id_instances)):
            sim = self.sims_dict.get_mashup_api_sim(mashup_id_instances[i], api_id_instances[i])
            sims.append(sim)

        examples = (np.array(mashup_id_instances),np.array(api_id_instances),np.array(sims))
        return examples

    def save_sth(self):
        self.sims_dict.save_sims_dict()

# TODO
class DHSR_noMF(DHSR_model):
    def get_name(self):
        return 'DHSR_noMF'

    def get_model(self):
        # Input Layer
        text_input = Input(shape=(self.args.sim_feature_size,), dtype='float32', name='text_input')

        predict_vector= Dense(self.args.final_MLP_layers[0], activation='relu')(text_input)

        for idx in range(len(self.args.final_MLP_layers))[1:]:   # 整合后再加上MLP？
            layer = Dense(self.args.final_MLP_layers[idx], activation='relu')# name="layer%d"  % idx
            predict_vector = layer(predict_vector)

        predict_vector = tf.keras.layers.Dropout(0.5)(predict_vector)    # 使用dropout?

        # Final prediction layer
        predict_vector = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(predict_vector)

        model = Model(inputs=text_input,outputs=predict_vector)
        return model

    def get_instances(self,mashup_id_instances, api_id_instances):
        sims=[]
        for i in range(len(mashup_id_instances)):
            sim = self.sims_dict.get_mashup_api_sim(mashup_id_instances[i], api_id_instances[i])
            sims.append(sim)

        returns=[]
        returns.append(sims)
        return np.array(returns)


class NCF_model(recommend_Model):
    def __init__(self):
        super(NCF_model, self).__init__()
        self.get_simple_name()
        self.set_paths()

    def get_simple_name(self):
        if not self.simple_name:
            self.simple_name = 'NCF_MLP{}_REGS{}_REG{}'.format(self.args.NCF_layers,self.args.NCF_reg_layers,
                                                               self.args.NCF_reg_mf).replace(',',' ')
        return self.simple_name

    def get_model(self):
        num_layer = len(self.layers)  # Number of layers in the MLP
        # Input variables
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')

        # Embedding layer
        MF_Embedding_User = Embedding(input_dim=self.num_users, output_dim=self.args.mf_embedding_dim, name='mf_embedding_user',
                                      embeddings_initializer=initializers.VarianceScaling(scale=0.01,distribution='normal'),
                                      embeddings_regularizer=l2(self.args.NCF_reg_mf), input_length=1) #

        MF_Embedding_Item = Embedding(input_dim=self.num_items, output_dim=self.args.mf_embedding_dim, name='mf_embedding_item',
                                      embeddings_initializer=initializers.VarianceScaling(scale=0.01,distribution='normal'),
                                      embeddings_regularizer=l2(self.args.NCF_reg_mf), input_length=1) #

        MLP_Embedding_User = Embedding(input_dim=self.num_users, output_dim=int(self.args.mf_fc_unit_nums[0] / 2), name="mlp_embedding_user",
                                       embeddings_initializer=initializers.VarianceScaling(scale=0.01,distribution='normal'),
                                       embeddings_regularizer=l2(self.args.NCF_reg_layers[0]), input_length=1) #

        MLP_Embedding_Item = Embedding(input_dim=self.num_items, output_dim=int(self.args.mf_fc_unit_nums[0] / 2), name='mlp_embedding_item',
                                       embeddings_initializer=initializers.VarianceScaling(scale=0.01,distribution='normal'),
                                       embeddings_regularizer=l2(self.args.NCF_reg_layers[0]), input_length=1) #

        # MF part
        mf_user_latent = tf.keras.layers.Flatten()(MF_Embedding_User(user_input))
        mf_item_latent = tf.keras.layers.Flatten()(MF_Embedding_Item(item_input))
        #   mf_vector = merge([mf_user_latent, mf_item_latent], mode='mul')  # element-wise multiply
        mf_vector=Multiply()([mf_user_latent, mf_item_latent])

        # MLP part
        mlp_user_latent = tf.keras.layers.Flatten()(MLP_Embedding_User(user_input))
        mlp_item_latent = tf.keras.layers.Flatten()(MLP_Embedding_Item(item_input))
        #   mlp_vector = merge([mlp_user_latent, mlp_item_latent], mode='concat')
        mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])

        for idx in range(1, num_layer):
            layer = Dense(self.args.mf_fc_unit_nums[idx],  activation='relu', name="layer%d" % idx) # kernel_regularizer=l2(reg_layers[idx]),
            mlp_vector = layer(mlp_vector)

        # Concatenate MF and MLP parts
        # mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
        # mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
        #   predict_vector = merge([mf_vector, mlp_vector], mode='concat')
        predict_vector = Concatenate()([mf_vector, mlp_vector])

        # Final prediction layer
        prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(predict_vector)

        model = Model(input=[user_input, item_input],output=prediction)
        return model

    def get_instances(self,mashup_id_instances, api_id_instances):
        examples = (np.array(mashup_id_instances),np.array(api_id_instances))
        return examples


