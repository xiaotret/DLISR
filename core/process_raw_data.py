# -*- coding:utf-8 -*-
import os

# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import WordPunctTokenizer,word_tokenize
import sys

import pandas as pd
from itertools import chain

from text_utils.encoding_padding_texts import encoding_padding
from Helpers.util import get_mashup_api_pair, write_mashup_api_pair, dict2list


# 源数据：包含mashup/api的各种信息，和mashup-api调用关系列表
# 基于df处理数据

class meta_data(object):
    # cur_md = None
    #
    # @classmethod
    # def initilize(cls, args):
    #     cls.cur_md = meta_data(args)
    #
    def __init__(self, args, min_api_num=2):  # 每个根目录一个对象
        self.args = args
        self.name = 'DATA_{}'.format(self.args.which_data)
        self.base_path = self.args.cur_data_dir
        self.mashup_raw_path = os.path.join(self.base_path, 'mashups.csv')  # /new_data/mashups.csv ***
        self.api_raw_path = os.path.join(self.base_path, 'apis.csv')
        self.processed_info_dir = os.path.join(self.base_path, 'processed_info')  # 所有生成文件的目录 /new_data/processed_info ***
        self.mashup_df_path = os.path.join(self.processed_info_dir, 'mashup.df')
        self.api_df_path = os.path.join(self.processed_info_dir, 'api.df')
        self.mashup_api_list_path = os.path.join(self.processed_info_dir, 'mashup_api_list.txt')
        self.statistics_path = os.path.join(self.processed_info_dir, 'statistics.csv')

        # mashup-api的调用矩阵，name形式，调用次数大于min_api_num的才纳入统计，可以在此基础上进行处理
        # 但 mashup.info中Related APIs字段存储了全部调用信息，跟min_api_num无关
        self.mashup_df,self.api_df = None, None
        self.des_pd, self.cate_pd = None, None # text和tag编码对象
        self.mashup_num = 0
        self.api_num = 0
        self.mashup_api_list = []  # mashup，api调用关系对，id 形式
        self.min_api_num = min_api_num
        self.process()
        if not os.path.exists(self.statistics_path):
            self.statistics()
        print('num of all apis:{},num of all mashups:{}'.format(self.api_num,self.mashup_num))

    def process(self):
        if not os.path.exists(self.processed_info_dir):
            os.makedirs(self.processed_info_dir)
        if not os.path.exists(self.api_df_path):
            mashup_df = self.process_raw_data(self.mashup_raw_path, 'mashup')
            api_df = self.process_raw_data(self.api_raw_path, 'api')
            mashup_df, api_df = self.filter_mashup_apis(mashup_df, api_df)
            self.encode_mashup_apis(mashup_df, api_df)
            self.encode_all_texts_cates() # 文本和tag分别编码！
            self.mashup_df.to_csv(self.mashup_df_path)
            self.api_df.to_csv(self.api_df_path)
        else:
            self.mashup_df = pd.read_csv(self.mashup_df_path)
            self.api_df = pd.read_csv(self.api_df_path)
            self.mashup_api_list = get_mashup_api_pair(self.mashup_api_list_path, 'list')
            print('load mashup_df,api_df and mashup_api_list, done!')
            self.encode_all_texts_cates()
        self.mashup_num = len(self.mashup_df) - 1  # 有效mashup(不含占位的)
        self.api_num = len(self.api_df) - 1 # 有效api(不含占位的)

    def process_raw_data(self, raw_data_path, mashup_api: str):
        """
        读取mashup/api的csv文件到df中，并处理各个字段
        :param raw_data_path: mashup/api的csv文件路径
        :param mashup_api: 'mashup' or 'api'
        :return:
        """

        # mashup_domains = ['Name', 'Description', 'Related_APIs', 'Categories', 'Company']
        # api_domains = ['Name', 'Description', 'Categories', 'API Provider']
        # 'Related_APIs','Categories'这些都是"['amazon-product-advertising', 'google-o3d']"形式，去除两边的""要eval()

        def process_categories(cate):
            # 有些标签有多个词汇，空格分隔. 现使用-连接:  "['amazon-product-advertising', 'google-o3d']"
            ans = []
            if not pd.isnull(cate):
                if type(cate) == str:  # 单个是['amazon-product-advertising'],不处理
                    cate = eval(cate.lower())
                for category in cate:
                    category_words = category.strip().split(' ')
                    if len(category_words) > 1:
                        category = '-'.join(category_words)
                    ans.append(category)
            return ans

        def process_Related_APIs(apis):
            ans = []
            if not pd.isnull(apis) and len(apis) > 0:  # []
                if type(apis) == str:
                    ans = eval(apis)
                else:
                    ans = apis
            return ans

        # 读取CSV文件到df并处理字段
        df = pd.read_csv(raw_data_path, encoding='unicode_escape')  # 'UTF-8'
        df['Name'] = df['Name'].map(lambda x: '-'.join(x.strip().lower().split() if x else ''), na_action='ignore')
        # Description 保持原始文本
        # TODO: nltk等分词，否则.,等符号在单词后，分不开
        #  df['Description'] = df['Description'].map(lambda x: nltk.word_tokenize(x.strip().lower()) if x else '',na_action='ignore')
        df['final_description'] = df['Description'].map(lambda x: x.strip().lower().split() if x else '', na_action='ignore')
        # df['final_description'] = df['final_description'].map(NLP_tool)
        df['Categories'] = df['Categories'].map(process_categories)
        if mashup_api == 'mashup':
            df['Related_APIs'] = df['Related_APIs'].map(process_Related_APIs)
        return df

    def filter_mashup_apis(self, mashup_df, api_df, filter_null_des=True, filter_null_cate=True, min_api_count=2):
        # 过滤描述或者类别字段为空的mashup/api
        if filter_null_des:
            mashup_df = mashup_df.dropna(subset=['Description'])
            api_df = api_df.dropna(subset=['Description'])
        if filter_null_cate:
            mashup_df = mashup_df.dropna(subset=['Categories'])
            api_df = api_df.dropna(subset=['Categories'])

        # 非空服务
        not_na_apis = set(api_df['Name'].values.tolist())
        if min_api_count > 0:
            # 过滤组件服务中的空api和有效组件数小于阈值的mashup
            mashup_df['valid_Related_APIs'] = mashup_df['Related_APIs'].map(
                lambda apis: list(filter(lambda x: x in not_na_apis, apis)))
            mashup_df['valid_Related_APIs_length'] = mashup_df['valid_Related_APIs'].map(len)
            mashup_df = mashup_df[mashup_df['valid_Related_APIs_length'] >= min_api_count]

        # 只保留非空且在mashup組件中出现过的服务 这么做,api几百;但如果不操作，api有2万多，测试时效率太低！
        appeared_useful_apis = set(list(chain(*mashup_df.valid_Related_APIs.tolist())))
        api_df = api_df[api_df['Name'].isin(appeared_useful_apis)]

        return mashup_df, api_df

    def encode_mashup_apis(self, mashup_df, api_df):
        # 对mashup/api编码，df中id是index；并获取mashup_api list
        # 新增占位的mashup/api,id=0
        mashup_df['id'] = range(1, len(mashup_df) + 1)
        api_df['id'] = range(1, len(api_df) + 1)
        api_name2id = {getattr(row, 'Name'): getattr(row, 'id') for row in api_df.itertuples()}
        self.mashup_api_list = dict2list({
            getattr(row, 'id'): [api_name2id[api_name] for api_name in getattr(row, 'valid_Related_APIs')]
            for row in mashup_df.itertuples()})

        api_df = api_df.append(
            [{'Name': 'placeholder-api', 'id': 0, 'Categories': [], 'Description': '', 'final_description': []}])
        mashup_df = mashup_df.append([{'Name': 'placeholder-mashup', 'id': 0, 'Related_APIs': [], 'Categories': [],
                                       'Description': '', 'final_description': []}])
        self.mashup_df = mashup_df.set_index(['id'])
        self.api_df = api_df.set_index(['id'])
        write_mashup_api_pair(self.mashup_api_list, self.mashup_api_list_path, 'list')
        return self.mashup_df, self.api_df, self.mashup_api_list

    # def encode_all_texts_cates(self):
    #     mashup_descriptions = self.mashup_df.final_description.tolist()  # [['word11','word12'],['word21','word22']...]
    #     api_descriptions = self.api_df.final_description.tolist()
    #     mashup_categories = self.mashup_df.Categories.tolist()  # [['cate11','cate12'],['cate21','cate22']...]
    #     api_categories = self.api_df.Categories.tolist()
    #     descriptions = mashup_descriptions + api_descriptions
    #     tags = mashup_categories + api_categories
    #     return descriptions, tags

    def encode_all_texts_cates(self):
        # 对于一个meta_data，对其文本和tag分别编码，返回编码结果(供CI使用，而baselines一般统一编码)
        mashup_descriptions = self.mashup_df.Description.tolist()  # ['sentence1','sentence2'...]
        api_descriptions = self.api_df.Description.tolist()
        descriptions = mashup_descriptions + api_descriptions
        self.des_pd = encoding_padding(descriptions, self.args, mode = 'text')
        # 结果严格按照mashup_df/api_df中的id顺序
        if 'unpadded_description' not in self.mashup_df.columns.tolist():
            self.mashup_df['unpadded_description'] = self.des_pd.get_texts_in_index(mashup_descriptions,padding=False) # TODO []?
            self.api_df['unpadded_description'] = self.des_pd.get_texts_in_index(api_descriptions,padding=False)
            self.mashup_df['padded_description'] = self.des_pd.get_texts_in_index(mashup_descriptions) # TODO []?
            self.api_df['padded_description'] = self.des_pd.get_texts_in_index(api_descriptions)

        mashup_categories = self.mashup_df['Categories'].map(lambda tokens:' '.join(eval(tokens))).tolist()  # [['cate11','cate12'],['cate21','cate22']...]
        api_categories = self.api_df['Categories'].map(lambda tokens:' '.join(eval(tokens))).tolist()
        categories = mashup_categories + api_categories
        self.cate_pd = encoding_padding(categories, self.args, mode = 'tag')

        if 'unpadded_categories' not in self.mashup_df.columns.tolist():
            self.mashup_df['unpadded_categories'] = self.cate_pd.get_texts_in_index(mashup_categories,padding=False)
            self.api_df['unpadded_categories'] = self.cate_pd.get_texts_in_index(api_categories,padding=False)
            self.mashup_df['padded_categories'] = self.cate_pd.get_texts_in_index(mashup_descriptions)
            self.api_df['padded_categories'] = self.cate_pd.get_texts_in_index(api_descriptions)

    def statistics(self):
        # 或者使用df.info() describe()查看
        api_num_per_mashup = sorted(self.mashup_df.valid_Related_APIs_length.tolist())  # 每个mashup构成的api数目
        mashup_description_lens = sorted(
            [len(m_des) for m_des in self.mashup_df.final_description.tolist()])  # if type(m_des)== str else 0
        mashup_categories_lens = sorted([len(m_cate) for m_cate in self.mashup_df.Categories.tolist()])
        api_description_lens = sorted([len(a_des) for a_des in self.api_df.final_description.tolist()])
        api_categories_lens = sorted([len(a_cate) for a_cate in self.api_df.Categories.tolist()])

        with open(self.statistics_path, 'w') as f:
            f.write('mashup num,{}\n'.format(self.mashup_num))
            f.write('api num,{}\n'.format(self.api_num))
            f.write('api_num_per_mashup,{}\n'.format(','.join(list(map(str, api_num_per_mashup)))))
            f.write('mashup_description_lens,{}\n'.format(','.join(list(map(str, mashup_description_lens)))))
            f.write('mashup_categories_lens,{}\n'.format(','.join(list(map(str, mashup_categories_lens)))))
            f.write('api_description_lens,{}\n'.format(','.join(list(map(str, api_description_lens)))))
            f.write('api_categories_lens,{}\n'.format(','.join(list(map(str, api_categories_lens)))))
        print('statistics,done!')


def NLP_tool(str_):  # 暂不处理
    return str_

# 没有安装NLTP时可以预先处理好再移植，之后不再调用NLTP
# english_stopwords = stopwords.words('english')  # 系统内置停用词
#
# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# def NLP_tool(raw_description, SpellCheck=False):  # 功能需进一步确认！！！
#     """
#     返回每个文本预处理后的词列表:
#     return [[],[]...]
#     """
#
#     """ 拼写检查
#     d=None
#     if SpellCheck:
#         d = enchant.Dict("en_US")
#     """
#
#     # st = LancasterStemmer()  # 词干分析器
#
#     words = []
#     """
#     line = re.sub(punctuaion, ' ', text)  # 不去标点，标点有一定含义
#     words= line.split()
#     """
#     for sentence in tokenizer.tokenize(raw_description):  # 分句再分词
#         # for word in WordPunctTokenizer().tokenize(sentence): #分词更严格，eg:top-rated会分开
#         for word in word_tokenize(sentence):
#             word=word.lower()
#             if word not in english_stopwords and word not in domain_stopwords:  # 非停用词
#                 """
#                 if SpellCheck and not d.check(word):#拼写错误，使用第一个选择替换？还是直接代替？
#                     word=d.suggest(word.lower())[0]
#                 """
#                 # word = st.stem(word)   词干化，词干在预训练中不存在怎么办? 所以暂不使用
#                 words.append(word)
#
#     return words
#
#
# def test_NLP(text):
#     for sentence in tokenizer.tokenize(text):  # 分句再分词
#         for word in WordPunctTokenizer().tokenize(sentence):
#             print(word + "\n")
