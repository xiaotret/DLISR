import os
import random
import pandas as pd
import numpy as np

from core import data_repository
from core.process_raw_data import meta_data
from Helpers.util import save_2D_list, list2dict, get_iterable_values
from mf.get_mf_embedding import MF


# 根据生成的训练测试集，生成一个dataset对象，里面实现了各种方法，更方便后续处理(train_data,test_data,his_mashup_ids的各种数据)
# 访问时，使用data_repository.get_ds() 访问当前使用的对象
# dataset.UV_obj 访问根据当前训练集生成的get_UV()对象，跟MF相关的数据

# 已划分的数据集对象,train/test dataset;以及供MF使用的UV对象(只使用training data)
class dataset(object):
    # crt_ds = None
    # MF_obj = None
    #
    # @classmethod
    # def set_cur_dataset(cls, dataset,args):  # 该类当前使用的dataset数据对象,必须先设置
    #     cls.crt_ds = dataset
    #     cls.MF_obj = dataset.set_MF_obj(args)

    def __init__(self, args,root_path, name, kcv_index=0):
        self.args = args
        self.kcv_index = kcv_index
        self.data_root = os.path.join(root_path, str(kcv_index))  # 存放数据的根路径
        self.train_df_path = os.path.join(self.data_root, 'train.df')
        self.test_df_path = os.path.join(self.data_root, 'test.df')
        # 数据名 eg: newScene_neg_{}_sltNum{}_com_{}_testCandi{}_KCV_{} 用于评测
        self.name = '{}_KCV{}'.format(name,kcv_index)
        self.train_data, self.test_data = None, None
        self.MF_obj = None

    def initialize(self, train_df=None, test_df=None):
        # 初始化train_df/test_df(设置并存储，或者读取)
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)
        if train_df is not None and test_df is not None:
            self.train_df = train_df
            self.test_df = test_df
            self.train_df.to_csv(self.train_df_path)
            self.test_df.to_csv(self.test_df_path)
        else:
            self.train_df = pd.read_csv(self.train_df_path)
            self.test_df = pd.read_csv(self.test_df_path)

        print('data train samples num:{}'.format(len(self.train_df)))
        self.set_others()  # 设置路径等
        self.set_train_test_data()  # 设置train test
        self.set_MF_obj()  # 训练集中的正例，供libRec等使用

    def set_others(self):
        # 在set_data()或read_data后设置
        self.his_mashup_ids = np.unique(self.train_df['mashup'].values)  # 训练mashup id的有序排列
        self.his_mashup_ids_set = set(self.his_mashup_ids)
        print('mashup num in training set :{}'.format(len(self.his_mashup_ids)))
        self.train_mashup_api_list = list(
            filter(lambda x: x[0] in self.his_mashup_ids_set, data_repository.get_md().mashup_api_list))
        self.train_mashup_api_dict = list2dict(self.train_mashup_api_list)

        # 模型随数据变化，所以存储在数据的文件夹下
        self.model_path = os.path.join(self.data_root, '{}')  # simple_model_name  CI路径
        self.new_best_epoch_path = os.path.join('{}', 'best_epoch.dat')  # model_dir,  .format(simple_model_name)
        self.new_model_para_path = os.path.join('{}', 'weights_{}.h5')  # model_dir, .format(simple_model_name, epoch)
        self.new_best_NDCG_path = os.path.join('{}', 'best_NDCG.dat')  # model_dir,  .format(simple_model_name)

    def set_train_test_data(self):
        """
        基于df，设置训练和测试数据(ID型),为训练测试模型提供统一的接口(在各模型文件中再把ID转化为需要的输入)
        :return: train_data和test_data都是字典类型
                具体的，list类型
        """
        self.train_df,self.test_df = self.train_df[:1000],self.test_df[:2] # 快速测试
        if self.train_data is None or self.test_data is None:
            if not self.args.pairwise:
                def expand_mashup_ids(row):
                    return [row['mashup']] * len(row['candidate_apis'])

                test_2D_mashup_ids = self.test_df.apply(expand_mashup_ids, axis=1).tolist()
                self.train_data = {
                    'mashup':self.train_df.mashup.tolist(),
                    'api':self.train_df.api.tolist(),
                    'label':self.train_df.label.tolist()
                }

                self.test_data = {
                    'mashup':test_2D_mashup_ids,
                    'api': get_iterable_values(self.test_df,'candidate_apis'),
                    'all_ground_api_ids':get_iterable_values(self.test_df,'all_ground_api_ids')
                }
                # 只有新场景下且需要slt apis时
                if self.args.data_mode == 'newScene' and self.args.need_slt_apis:
                    self.train_data['slt_apis'] = get_iterable_values(self.train_df,'slt_apis')
                    self.test_data['slt_apis'] = get_iterable_values(self.test_df,'slt_apis')

            # TODO 还没改成df
            else:  # pairwise型的训练数据，根据pointwise型的转化
                dict_pos = {}
                dict_neg = {}
                for index in range(len(self.train_mashup_id_list)):
                    mashup_id = self.train_mashup_id_list[index]
                    api_id = self.train_api_id_list[index]
                    slt_api_ids = self.slt_api_ids_instances[index]

                    key = (mashup_id, tuple(
                        slt_api_ids)) if self.args.data_mode == 'newScene' and self.args.need_slt_apis else mashup_id

                    if key not in dict_pos.keys():
                        dict_pos[key] = []
                    if key not in dict_neg.keys():
                        dict_neg[key] = []
                    if self.train_labels[index] == 1:
                        dict_pos[key].append(api_id)  # 可以包含多个正例
                    else:
                        dict_neg[key].append(api_id)

                train_mashup_id_list, train_pos_api_id_list, slt_api_ids_instances, train_neg_api_id_list = [], [], [], []
                for key in dict_pos.keys():
                    # assert len()
                    pos_api_ids = dict_pos[key] * self.args.num_negatives
                    train_pos_api_id_list.extend(pos_api_ids)
                    neg_api_ids = dict_neg[key]
                    train_neg_api_id_list.extend(neg_api_ids)
                    pair_num = len(neg_api_ids)
                    train_mashup_ids = [key[0]] * pair_num
                    train_mashup_id_list.extend(train_mashup_ids)
                    if self.args.data_mode == 'newScene' and self.args.need_slt_apis:
                        slt_api_ids = list(key[1])
                        for i in range(pair_num):
                            slt_api_ids_instances.append(slt_api_ids)
                train_labels = [1] * len(train_mashup_id_list)  # 随便设，占位而已

                self.train_data = {
                    'mashup':train_mashup_id_list,
                    'api':train_pos_api_id_list,
                    'neg_api':train_neg_api_id_list,
                    'label':train_labels
                }
                self.test_data = {
                    'mashup':self.test_mashup_id_list,
                    'api':self.test_api_id_list,
                    'all_ground_api_ids':self.grounds
                }
                # 只有新场景下且需要slt apis时
                if self.args.data_mode == 'newScene' and self.args.need_slt_apis:
                    self.train_data['slt_apis'] = slt_api_ids_instances,
                    self.test_data['slt_apis'] = self.test_slt_ids
        return self.train_data, self.test_data

    def set_MF_obj(self):  # 'pmf','BPR','listRank','Node2vec'
        # 根据train mashups 将全部mashup_api_list划分，得到训练数据集，进行MF
        if self.MF_obj is None:
            # UV对象中mashup id等也是按顺序排列，跟his_mashup_ids 一样
            self.MF_obj = MF(self.data_root, self.args.mf_mode)
            self.MF_obj.get_mf_embedding(self.train_mashup_api_list,self.args.mf_mode) # 初始化
        return self.MF_obj

    # 下面的几个方法是转化训练，测试集，方便测试其他模型
    def reduct(self, data):
        Mid_Aid_lables = {}
        _zip = zip(data[0], data[1])
        for index, key in enumerate(_zip):
            if tuple(key) not in set(Mid_Aid_lables.keys()):
                Mid_Aid_lables[key] = data[-1][index]
        _1, _2 = zip(*Mid_Aid_lables.keys())
        return (list(_1), list(_2), list(Mid_Aid_lables.values()))

    def transfer(self):
        # 将无slt apis的含重复数据去重   'newScene'且need_slt_apis=False时
        if self.train_data is None:
            self.set_train_test_data()
        print('before reduction, train samples:{}'.format(len(self.train_data[0])))
        self.train_data = self.reduct(self.train_data)
        print('after reduction, train samples:{}'.format(len(self.train_data[0])))

        self.test_data_no_reduct = self.test_data  # 只评价去重后的，样本太少，不太公平
        test_mashup_ids_set = set()
        test_mashup_id_list, test_api_id_list, grounds = [], [], []
        for index, test_mashup_ids in enumerate(self.test_mashup_id_list):
            if test_mashup_ids[0] not in test_mashup_ids_set:
                test_mashup_ids_set.add(test_mashup_ids[0])
                test_mashup_id_list.append(test_mashup_ids)
                test_api_id_list.append(self.test_api_id_list[index])
                grounds.append(self.grounds[index])
        print('before reduction, test samples:{}'.format(len(self.test_data[0])))
        self.test_data = test_mashup_id_list, test_api_id_list, grounds
        print('after reduction, test samples:{}'.format(len(self.test_data[0])))
        print('remove_,done!')

    def transfer_false_test_DHSR(self, if_reduct_train=False):
        # 为了新场景下让DHSR/SVD等可以work，假设它们可以根据用户选择实时训练，把测试集中已选择的服务，加入训练集
        # 使用跟我们的模型类似的self.train_data,self.test_data
        # 把测试集分为几种情况： 选择一个服务的(只有一个作为正例)；2个的；3个的。分别跟训练集整合在一起，作为综合训练集

        # 训练集不做改变，不同已选，相同的正例，多次出现也无所谓，跟MISR的数据保持一致

        if if_reduct_train:  # 约减原始训练集中的大量重复信息，否则显得测试集的伪训练集很小
            self.train_mashup_id_list, self.train_api_id_list, self.train_labels = self.reduct(self.train_data)

        self.train_data, self.test_data = [], []  # 改变格式，按照测试集已选的数目，生成几个不同的训练和测试
        all_apis = {api_id for api_id in range(meta_data.api_num)}
        num_negatives = self.args.num_negatives

        # 把测试集的数据转化为训练集，按照已选服务个数划分
        # 分别训练测试，得到1,2,3场景下的指标
        def certain_slt_num_split(slt_num):
            set_ = set()  # 存储某个mashup，某个长度已选的数据的集合
            train_mashup_id_list, train_api_id_list, labels = list(self.train_mashup_id_list), list(
                self.train_api_id_list), list(self.train_labels)
            test_mashup_id_list, test_api_id_list, grounds = [], [], []
            for index, test_mashup_ids in enumerate(self.test_mashup_id_list):
                m_id, slt_api_ids = test_mashup_ids[0], self.test_slt_ids[index]
                if len(slt_api_ids) == slt_num and (m_id, len(slt_api_ids)) not in set_:
                    train_mashup_id_list.extend([m_id] * slt_num * (num_negatives + 1))  # 测试集中已选的服务作为正例,还有负例

                    train_api_id_list.extend(slt_api_ids)
                    neg_api_ids = list(all_apis - set(slt_api_ids))
                    random.shuffle(neg_api_ids)
                    neg_api_ids = neg_api_ids[:slt_num * num_negatives]
                    train_api_id_list.extend(neg_api_ids)

                    labels.extend([1] * slt_num)
                    labels.extend([0] * slt_num * num_negatives)

                    # 同时也需要测试，跟原来格式相同
                    test_mashup_id_list.append(test_mashup_ids)
                    test_api_id_list.append(self.test_api_id_list[index])
                    grounds.append(self.grounds[index])

            self.train_data.append((train_mashup_id_list, train_api_id_list, labels))
            self.test_data.append((test_mashup_id_list, test_api_id_list, grounds))

        for i in range(1, self.args.slt_item_num + 1):
            print('slt_num:', i)
            print('before, train samples:{}'.format(len(self.train_mashup_id_list)))
            certain_slt_num_split(i)
            print('after, train samples:{},{}'.format(len(self.train_data[-1][0]), len(self.train_data[-1][-1])))
            print('after, test samples:{}'.format(len(self.test_data[i - 1][0])))
        print('transfer for DHSR,done!')
        return self.train_data, self.test_data

    def transfer_false_test_MF(self):
        # 只需要返回一个train_mashup_api_list
        # 把测试集的数据转化为训练集，按照已选服务个数划分
        # 分别训练测试，得到1,2,3场景下的指标

        # 正例训练集
        train_mashup_id_list, train_api_id_list = [], []
        Mid_Aid_set = set()
        _zip = zip(self.train_data[0], self.train_data[1])
        train_labels = self.train_data[-1]
        for index, Mid_Aid_pair in enumerate(_zip):
            if train_labels[index] and tuple(Mid_Aid_pair) not in Mid_Aid_set:  # 正例且之前未出现过
                train_mashup_id_list.append(Mid_Aid_pair[0])
                train_api_id_list.append(Mid_Aid_pair[1])

        def certain_slt_num_split(train_mashup_id_list, train_api_id_list, slt_num):
            # 测试集
            set_ = set()  # 存储某个mashup，某个长度已选的数据的集合
            test_mashup_id_list, test_api_id_list, grounds = [], [], []
            for index, test_mashup_ids in enumerate(self.test_mashup_id_list):
                m_id, slt_api_ids = test_mashup_ids[0], self.test_slt_ids[index]
                if len(slt_api_ids) == slt_num and (m_id, len(slt_api_ids)) not in set_:
                    train_mashup_id_list.extend([m_id] * slt_num)  # 测试集中已选的服务作为正例,还有负例
                    train_api_id_list.extend(slt_api_ids)

                    # 同时也需要测试，跟原来格式相同
                    test_mashup_id_list.append(test_mashup_ids)
                    test_api_id_list.append(self.test_api_id_list[index])
                    grounds.append(self.grounds[index])

            self.train_data.append(list(zip(train_mashup_id_list, train_api_id_list)))  # 供get_U_V使用
            self.test_data.append((test_mashup_id_list, test_api_id_list, grounds))

        self.train_data, self.test_data = [], []  # 改变格式，按照测试集已选的数目，生成几个不同的训练和测试
        for i in range(1, self.args.slt_item_num + 1):
            print('slt_num:', i)
            print('before, train samples:{}'.format(len(self.train_mashup_id_list)))
            certain_slt_num_split(list(train_mashup_id_list), list(train_api_id_list), i)
            print('after, train samples:{}'.format(len(self.train_data[-1])))
            true_train_set_path = os.path.join(self.data_root, 'train_set_MF_{}.data'.format(i))
            save_2D_list(true_train_set_path, self.train_data[i - 1])  # 把训练集(加上了某个长度的测试集)存起来，java处理
        print('transfer for MF,done!')
        return self.train_data, self.test_data

    def get_few_samples(self, train_num, test_num=32):
        """
        测试模型时只利用一小部分样本做训练和测试
        :param train_num:
        :return:
        """
        if self.train_data is None:
            self.set_train_test_data()
        if self.args.data_mode == 'newScene' and self.args.need_slt_apis:
            return ((self.train_data[0][:train_num], self.train_data[1][:train_num], self.train_data[2][:train_num],
                     self.train_data[3][:train_num]),
                    (self.test_data[0][:test_num], self.test_data[1][:test_num], self.test_data[2][:test_num],
                     self.test_data[3][:test_num]))
        else:
            return ((self.train_data[0][:train_num], self.train_data[1][:train_num], self.train_data[2][:train_num]),
                    (self.test_data[0][:test_num], self.test_data[1][:test_num], self.test_data[2][:test_num]))
