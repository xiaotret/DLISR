import os
import pickle
from tensorflow.keras.models import Model


class get_midMLP_feature(object):

    # 得到并存储CI,NI模型的中间层结果，供topMLP复用
    def __init__(self, recommend_model, model, if_explict=False):
        self.recommend_model = recommend_model
        self.whole_model = model  # 整个模型
        self.explict = if_explict  # NI显式模型，旧场景
        self.model = None  # 中间模型
        self.flag = False  # 有sample新求的话，训练和测试结束后更新一次
        self.midMLP_fea_path = os.path.join(recommend_model.model_dir, 'midMLP_feature.dat')
        self.input2fea = None  # 存储输入到中间输出的字典,训练和测试集全部的

    def get_midMLP_feature(self, layer_name,data,pairwise_train_phase_flag=False):
        """
        根据输入得到最上层MLP中间结果，分批，训练和batch测试时多次调用
        layer_name:
        CI: 'text_tag_feature_extracter' ;
        隐式NI: 'implict_dense_{}'.format(len()-1)
        显式NI: 'explict'/'cor'_dense_{}'.format(len()-1)
        :return:
        """
        mashup_id_instances = data.get('mashup')
        api_id_instances = data.get('api')
        slt_api_ids_instances = data.get('slt_apis')
        num = len(mashup_id_instances)

        if self.input2fea is None:  # 第一次用
            if os.path.exists(self.midMLP_fea_path):
                with open(self.midMLP_fea_path, 'rb') as f:
                    self.input2fea = pickle.load(f)
                    print('load existed midMLP_fea_file!')
            else:
                print('no existed midMLP_fea_file!')
                self.input2fea = {}  # 输入到feature，不用numpy因为测试样例格式特殊
                self.flag = True

        if slt_api_ids_instances is not None:  # attention模型
            if (mashup_id_instances[0], api_id_instances[0], tuple(slt_api_ids_instances[0])) not in self.input2fea:
                # 成批的(训练集或一个batch的测试集)，一个不在的话全部没算过
                self.flag = True
                if self.model is None:
                    self.model = Model(inputs=[*self.whole_model.inputs],
                                       outputs=[self.whole_model.get_layer(layer_name).output])
                instances_dict = self.recommend_model.get_instances(data, pairwise_train_phase_flag=pairwise_train_phase_flag)
                midMLP_features = self.model.predict(instances_dict, verbose=0)  # 现求的二维numpy
                for index in range(num):
                    key = (mashup_id_instances[index], api_id_instances[index], tuple(slt_api_ids_instances[index]))
                    self.input2fea[key] = midMLP_features[index]
            else:
                midMLP_features = \
                    [self.input2fea[(mashup_id_instances[index], api_id_instances[index], tuple(slt_api_ids_instances[index]))]
                                   for index in range(num)]
        else:
            midMLP_features = []
            for index in range(num): # 在新场景，不使用已选服务时，样本大量重复(相同的m_id,a_id)，这么写提速
                key = (mashup_id_instances[index], api_id_instances[index])
                if key not in self.input2fea:
                    self.flag = True
                    if self.model is None:
                        if not self.explict:
                            self.model = Model(inputs=[*self.whole_model.inputs],
                                               outputs=[self.whole_model.get_layer(layer_name).output])
                        else: # 针对显式交互，特殊
                            self.model = Model(input=self.whole_model.input,
                                               output=self.whole_model.get_layer(layer_name).output)
                    a_instance_dict = self.recommend_model.get_instances([key[0]], [key[1]],
                                                                         pairwise_train_phase_flag=pairwise_train_phase_flag)
                    if not self.explict: # TODO
                        a_midMLP_feature = self.model.predict(a_instance_dict, verbose=0)[0]  # 现求的二维numpy
                    else:
                        a_midMLP_feature = self.model.predict(a_instance_dict, verbose=0)

                    self.input2fea[key] = a_midMLP_feature
                else:
                    a_midMLP_feature = self.input2fea[key]
                midMLP_features.append(a_midMLP_feature)
        return midMLP_features

    def save_sth(self):  # 结束之后存储中间feature
        if self.flag:
            with open(self.midMLP_fea_path, 'wb+') as f:
                pickle.dump(self.input2fea, f)
            print('save midMLP_fea_file,done!')
