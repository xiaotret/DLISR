import random
from gensim.corpora import Dictionary
from gensim.models import HdpModel, TfidfModel, LdaModel
import numpy as np

from core.dataset import dataset
from core import data_repository
from Helpers.util import get_iterable_values


class gensim_data(object):
    def __init__(self, mashup_descriptions,mashup_categories,api_descriptions,api_categories,mashup_only=False,tag_times=2,strict_train=False):
        """
        使用gensim处理mashup和api的text和tag，得到特征表示，主题表示等
        :param tag_times: 编码时是否使用tag，以及tag的次数
        :param mashup_only: 是否只使用mashup信息: 用于LDA对mashup聚类
        :param strict_train: 是否仅使用训练集的信息
        """
        self.strict_train = strict_train
        self.mashup_only = mashup_only
        self.num_topics = 0
        self.model = None  # 处理文本的模型
        self.mashup_features,self.api_features = None,None # 使用模型处理文本得到稀疏特征向量
        self._mashup_features,self._api_features = None,None  # 处理后的密集特征向量
        self.mashup_topics,self.api_topics = None,None  # 文本最高的N个topic

        def initialize():
            # 整合text和tag信息：一个mashup/api的信息整合在一起，一行
            if tag_times > 0:
                assert len(mashup_descriptions) == len(mashup_categories)
                self.mashup_dow = []
                for i in range(len(mashup_descriptions)):
                    # 直接将文本和tag拼接，是否有更好的方法
                    self.mashup_dow.append(mashup_descriptions[i] + mashup_categories[i] * tag_times)
            else:
                self.mashup_dow = mashup_descriptions

            if tag_times > 0:
                assert len(api_descriptions) == len(api_categories)
                self.api_dow = []
                for i in range(len(api_descriptions)):
                    self.api_dow.append(api_descriptions[i] + api_categories[i] * tag_times)
            else:
                self.api_dow = api_descriptions

            if self.strict_train:
                # 训练用的mashup，api的编码
                self.train_mashup_dow = [self.mashup_dow[m_id] for m_id in data_repository.get_ds().his_mashup_ids]
                self.dct = Dictionary(self.train_mashup_dow)
                self.train_mashup_dow = [self.dct.doc2bow(mashup_info) for mashup_info in
                                         self.train_mashup_dow]  # 词id-数目
            else:
                self.dct = Dictionary(self.mashup_dow + self.api_dow)

            # 为每个mashup/api计算feature
            self.mashup_dow = [self.dct.doc2bow(mashup_info) for mashup_info in self.mashup_dow]  # 所有mashup文本的词id-数目
            self.api_dow = [self.dct.doc2bow(api_info) for api_info in self.api_dow]
            # print('len of self.mashup_dow,self.api_dow:{},{}'.format(len(self.mashup_dow),len (self.api_dow)))

        initialize()

    def encode(self,docs):
        # 基于建立的词典，对文本编码
        return list(map(self.dct.doc2idx,docs))

    def get_feas(self,docs):
        # 编码并获得特征向量
        dows = list(map(self.dct.doc2idx,docs))
        feas = [self.model[dow] for dow in dows]
        return feas

    def get_all_encoded_comments(self):
        self.unpadded_encoded_mashup_texts = self.encode(get_iterable_values(data_repository.get_md().mashup_df,'final_description'))
        self.unpadded_encoded_mashup_tags = self.encode(get_iterable_values(data_repository.get_md().mashup_df,'Categories'))
        self.unpadded_encoded_api_texts = self.encode(get_iterable_values(data_repository.get_md().api_df,'final_description'))
        self.unpadded_encoded_api_tags = self.encode(get_iterable_values(data_repository.get_md().api_df,'Categories'))

    # 只关注词在文本中是否出现过，二进制，用于计算cos和jaccard
    def get_binary_v(self):
        dict_size = len(self.dct)
        mashup_binary_matrix = np.zeros((data_repository.get_md().mashup_num+1, dict_size))
        api_binary_matrix = np.zeros((data_repository.get_md().api_num+1, dict_size))
        mashup_words_list = []  # 每个mashup中所有出现过的词
        api_words_list = []
        for id in range(data_repository.get_md().mashup_num+1):
            temp_words_list, _ = zip(*self.mashup_dow[id])
            mashup_words_list.append(temp_words_list)
            for j in temp_words_list:  # 出现的词汇index
                mashup_binary_matrix[id][j] = 1.0

        for id in range(data_repository.get_md().api_num+1):
            temp_words_list, _ = zip(*self.api_dow[id])
            api_words_list.append(temp_words_list)
            for j in temp_words_list:  # 出现的词汇index
                api_binary_matrix[id][j] = 1.0
        return mashup_binary_matrix, api_binary_matrix, mashup_words_list, api_words_list

    def model_pcs(self, model_name, LDA_topic_num=None):
        # 模型处理，返回mashup和api的特征:对同一个语料，可以先后使用不同的模型处理
        # hdp结果形式：[(0, 0.032271167132309014),(1, 0.02362695056720504)]
        if self.mashup_only:
            if self.strict_train:
                train_corpus = self.train_mashup_dow
            else:
                train_corpus = self.mashup_dow
        else:
            if self.strict_train:
                train_corpus = self.train_mashup_dow + self.api_dow
            else:
                train_corpus = self.mashup_dow + self.api_dow

        if model_name == 'HDP':
            self.model = HdpModel(train_corpus, self.dct)
            self.num_topics = self.model.get_topics().shape[0]
            print('num_topics', self.num_topics)
        elif model_name == 'TF_IDF':
            self.model = TfidfModel(train_corpus)
            self.num_topics = len(self.dct)
        elif model_name == 'LDA':
            if LDA_topic_num is None:
                self.model = LdaModel(train_corpus)
            else:
                self.model = LdaModel(train_corpus, num_topics=LDA_topic_num)
            self.num_topics = self.model.get_topics().shape[0]
        else:
            raise ValueError('wrong gensim_model name!')

        # 使用模型处理文本得到稀疏特征向量，再转化为标准的np格式(每个topic上都有)
        # *** 由于mashup_dow和api_dow默认是全部mashup/api的文本，所以得到的特征列表用全局性的id索引即可 ***
        self.mashup_features = [self.model[mashup_info] for mashup_info in self.mashup_dow]  # 每个mashup和api的feature
        self.api_features = [self.model[api_info] for api_info in self.api_dow]
        self._mashup_features = np.zeros((data_repository.get_md().mashup_num, self.num_topics))
        self._api_features = np.zeros((data_repository.get_md().api_num, self.num_topics))
        for i in range(data_repository.get_md().mashup_num):  # 部分维度有值，需要转化成规范array
            for index, value in self.mashup_features[i]:
                self._mashup_features[i][index] = value
        for i in range(data_repository.get_md().api_num):
            for index, value in self.api_features[i]:
                self._api_features[i][index] = value
        return self._mashup_features, self._api_features

    def get_topTopics(self, topTopicNum=3):  # 选取概率最高的topK个主题 [(),(),...]
        mashup_topics = []
        api_topics = []
        for index in range(data_repository.get_md().mashup_num):
            sorted_mashup_feature = sorted(self.mashup_features[index], key=lambda x: x[1], reverse=True)
            try:
                topic_indexes, _ = zip(*sorted_mashup_feature)
            except:
                # 有时mashup_bow非空，但是mashup_feature为空
                topic_indexes = random.sample(range(data_repository.get_md().mashup_num), topTopicNum)
            num = min(len(topic_indexes), topTopicNum)
            mashup_topics.append(topic_indexes[:num])
        for index in range(data_repository.get_md().api_num):
            sorted_api_feature = sorted(self.api_features[index], key=lambda x: x[1], reverse=True)
            try:
                topic_indexes, _ = zip(*sorted_api_feature)
            except:
                topic_indexes = random.sample(range(data_repository.get_md().api_num), topTopicNum)
            num = min(len(topic_indexes), topTopicNum)
            api_topics.append(topic_indexes[:num])
        return mashup_topics, api_topics


def get_default_gd(tag_times=2, mashup_only=False,strict_train=False):
    # 对mashup和api的文本+tag，统一处理
    gd = gensim_data(get_iterable_values(data_repository.get_md().mashup_df,'final_description'),
                     get_iterable_values(data_repository.get_md().mashup_df,'Categories'),
                     get_iterable_values(data_repository.get_md().api_df,'final_description'),
                     get_iterable_values(data_repository.get_md().api_df,'Categories'),
                     tag_times = tag_times,
                     mashup_only = mashup_only,
                     strict_train=strict_train)  # 调整tag出现的次数
    return gd
