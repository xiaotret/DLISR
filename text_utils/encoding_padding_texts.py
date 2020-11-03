# -*- coding:utf-8 -*-
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class encoding_padding(object):
    """
    文本编码类：根据词频建立词典
    """
    def __init__(self,descriptions,args,mode='text'):
        self.tokenizer = None
        self.word2index =None
        self.max_len = args.MAX_SEQUENCE_LENGTH if mode=='text' else args.MAX_TAGS_NUM
        self.args = args
        self.process_text(descriptions)

    def process_text(self,descriptions):
        """
        process descriptions
        默认按照文本中词频建立词典   0空白 1 最高频词汇 ....！
        :return:
        """

        print('Found %s texts.' % len(descriptions))
        filters= self.args.keras_filter_puncs if self.args.remove_punctuation else '' # 是否过滤标点
        self.tokenizer = Tokenizer(filters=filters, num_words=self.args.MAX_NUM_WORDS)  # 声明最大长度，默认使用词频***
        for i,description in enumerate(descriptions): # 处理占位的空文档
            if isinstance(description,float):
                descriptions[i] = ''
        self.tokenizer.fit_on_texts(descriptions)

        # 字典：将单词（字符串）映射为索引
        self.word2index = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(self.word2index))

        # 截断
        self.word2index = dict(filter(lambda x:x[1]<=self.args.MAX_NUM_WORDS,self.word2index.items()))
        # import json
        # jsObj = json.dumps(self.word2index)
        # with open('jsonFile.json', 'w') as f:
        #     f.write(jsObj)

        """
        with open('../data/word2index','w',encoding='utf-8') as f:
            for word,index in self.word2index.items():
                f.write('{} {}\n'.format(word,index))
        np.savetxt('../data/keras_encoding_texts',self.texts_in_index_padding,fmt='%d')
        print('save keras_encoding_texts,done!')
        """

    def human_encode(self,text):
        result = [0] * self.max_len
        size = len(text)
        _sub = self.max_len - size
        for i in list(range(size))[::-1]:  # 反向开始变换
            word_index = self.word2index.get(text[i])
            if word_index is not None:
                result[_sub + i] = word_index
        return result

    def get_texts_in_index(self, texts, manner = 'keras_setting',padding =  True):
        """
        得到多个文本的词汇index编码形式
        :param texts: 输入的文本列表, word的二维列表
        :param manner: 选择编码方式：keras内置函数还是利用word2indedx字典手动编码
        :param padding: 是否padding
        :return:
        """
        if manner=='keras_setting':
            res = self.tokenizer.texts_to_sequences(texts)
            if padding:
                res = pad_sequences(res, maxlen=self.max_len).tolist()
            # print(type(res))
            return res

        elif manner=='self_padding':
            if padding:
                return list(map(self.human_encode(),texts))
            else:
                return [[self.word2index.get(word) for word in text]for text in texts]
        else:
            raise ValueError('wrong manner!')
        return 0


