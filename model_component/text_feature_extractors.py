import tensorflow as tf
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Concatenate, MaxPooling2D, LSTM, Bidirectional
from tensorflow.keras.layers import Input, Conv2D, Embedding
from tensorflow.keras.models import Model


def bert_extracter_from_texts():
    pass


def textCNN_feature_extracter_from_texts(embedded_sequences,args):
    """
    对embedding后的矩阵做textCNN处理提取特征
    :param embedded_sequences:
    :return:
    """
    filtersize_list = [3, 4, 5]
    number_of_filters_per_filtersize = args.textCNN_channels  # 跟50D接近   #[128,128,128]
    pool_length_list = [2, 2, 2]

    conv_list = []
    for index, filtersize in enumerate(filtersize_list):
        nb_filter = number_of_filters_per_filtersize[index]
        pool_length = pool_length_list[index]
        conv = Conv2D(nb_filter, kernel_size=(filtersize, args.embedding_dim), activation='relu')(
            embedded_sequences)
        pool = MaxPooling2D(pool_size=(pool_length, 1))(conv)
        print('a feature map size:', pool)
        flatten = tf.keras.layers.Flatten()(pool)
        conv_list.append(flatten)

    if (len(filtersize_list) > 1):
        out = Concatenate(axis=-1)(conv_list)
    else:
        out = conv_list[0]
    return out


def LSTM_feature_extracter_from_texts(embedded_sequences,args):
    out = Bidirectional(LSTM(args.LSTM_dim))(embedded_sequences)
    # out = LSTM(new_Para.param.LSTM_dim)(embedded_sequences)
    return out


def SDAE_feature_extracter_from_texts():
    pass


fixed_vector_modes = ['HDP','Bert']
def vector_feature_extracter_from_texts(mashup_api:str, features):
    # 使用固定的文本特征初始化embedding层，HDP和Bert可用
    ID_input = Input(shape=(1,), dtype='int32')
    text_embedding_layer = Embedding(len(features), len(features[0]),
                                            embeddings_initializer=Constant(features),
                                            mask_zero=False, input_length=1,
                                            trainable=False, name=mashup_api+'_text_embedding_layer')
    x = text_embedding_layer(ID_input)
    feature_extracter = Model(ID_input, x, name=mashup_api+'_text_feature_extracter')
    return feature_extracter
