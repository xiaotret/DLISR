import pathlib
import sys
import os
import argparse
import tensorflow as tf

project_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)
sys.path.append(project_path)

# 使用多个gpu时，报错：E tensorflow/stream_executor/cuda/cuda_dnn.cc:329] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
# 问题出在RTX2070/2080显卡的显存分配问题上，将 GPU 的显存使用策略设置为 “仅在需要时申请显存空间”
# gpus= tf.config.list_physical_devices('GPU')
# print(gpus)
# tf.config.experimental.set_memory_growth(device=gpus[-1], enable=True)
# 但还是有问题，改为只使用一个gpu!
# https://github.com/tensorflow/tensorflow/issues/24496#
# 可以用os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from core.path_config import new_data_dir,old_data_dir
from core import data_repository
from core.process_raw_data import meta_data
from core.split_data import KCV_dataset_generater
from core.dataset import dataset
# from core.run_models import baselines, CI_NI_fineTuning
from core.run_models import CI_NI_fineTuning


punctuations = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n' # '[<>/\s+\.\!\/_,;:$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+'
stop_words = set(['!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'])

parser = argparse.ArgumentParser()
# --:可选 -short choices

# 1.1 meta-data相关：文本+category
parser.add_argument("--remove_punctuation", type=bool, default=True, help="文本是否去除标点")
parser.add_argument("--keras_filter_puncs", type=str, default=punctuations, help="keras编码时默认过滤的token，一般是标点")
parser.add_argument("--embedding_name", type=str, default='glove', help="word embedding type")
parser.add_argument("--embedding_dim", type=int, default=50, help="word embedding dim")
parser.add_argument("--MAX_SEQUENCE_LENGTH", type=int, default=100, help="最大文本长度")
parser.add_argument("--MAX_BERT_SEQUENCE_LENGTH", type=int, default=150, help="最大文本长度")
parser.add_argument("--MAX_TAGS_NUM", type=int, default=20, help="最大tag长度")
parser.add_argument("--MAX_NUM_WORDS", type=int, default=40000, help="最大词典大小")
# parser.add_argument("--Category_type", type=str, default='all', choices=['all', 'first', 'second'],
#                     help="使用哪些category")

# 1.2 数据划分相关
parser.add_argument("--which_data", type=str, default='new',help="使用哪个数据集")
parser.add_argument("--cur_data_dir", type=str, default='',help="当前使用数据目录")
parser.add_argument("--data_mode", type=str, default='newScene', choices=['newScene', 'oldScene'],help="模型针对的数据场景") # 新场景就是多种已选服务的组合
parser.add_argument("--need_slt_apis", type=bool, default=True, help="样本中是否需要已选择服务") # 新数据也可以用在旧模型中，只需要设为false即可***
parser.add_argument("--num_negatives", type=int, default=12, help="负样本-正样本比例")
# -- 新场景数据集使用
parser.add_argument("--slt_item_num", type=int, default=3, help="已选服务组合的最大的size")
parser.add_argument("--combination_num", type=int, default=6, help="已选服务组合的最大数目")
parser.add_argument("--test_candidates_nums", type=str, default='all', help="待测服务个数")


# 2.Model
# -CI
# --文本特征提取
parser.add_argument("--text_extracter_mode", type=str, default='inception', choices=['inception', 'LSTM', 'textCNN','bert'],
                    help="文本特征提取器")
parser.add_argument("--inception_channels", type=list, default=[10, 10, 10, 20, 10], help="inception中各种通道数")
parser.add_argument("--inception_pooling", type=str, default='global_avg',choices=['global_max', 'max', 'global_avg', 'none'])
parser.add_argument("--if_inception_MLP", type=bool, default=True, help="textCNN中各种通道数")
parser.add_argument("--inception_fc_unit_nums", type=list, default=[100, 50], help="textCNN中各种通道数")
parser.add_argument("--inception_MLP_dropout", type=bool, default=True, help="textCNN中各种通道数")
parser.add_argument("--inception_MLP_BN", type=bool, default=False, help="textCNN中各种通道数")
parser.add_argument("--textCNN_channels", type=list, default=[20, 20, 20], help="textCNN中各种通道数")
parser.add_argument("--LSTM_dim", type=int, default=25, help="LSTM unit num")
parser.add_argument("--frozen_bert", type=bool, default=True, help="是否冻住BERT")

# --整合text和category的特征
parser.add_argument("--merge_manner", type=str, default='direct_merge', choices=['direct_merge', 'final_merge'],
                    help="text和category特征直接拼接(direct)还是MLP交互后再拼接(final)")
parser.add_argument("--text_fc_unit_nums", type=list, default=[100, 50], help="final_merge,text交互时MLP结构")
parser.add_argument("--tag_fc_unit_nums", type=list, default=[100, 50], help="final_merge,tag交互时MLP结构")
parser.add_argument("--content_fc_unit_nums", type=list, default=[200, 100, 50], help="内容交互时最终MLP结构") # [100, 50] [256,64,16,8] [1024,256,64]

parser.add_argument("--model_mode", type=str, default='CI',choices=['CI', 'LR_PNCF'],help= 'CI部分特征交互方式') # LR_PNCF是对各种特征做哈达玛交互
parser.add_argument("--CI_handle_slt_apis_mode", type=str, default='attention',choices=['attention', 'full_concate', 'average', ''],
                    help= 'CI中处理多个已选服务的方式')
parser.add_argument("--simple_CI_slt_mode", type=str, default='att',choices=['att', 'ful', 'ave', ''],
                    help= 'CI中处理多个已选服务的方式')

# -NI
parser.add_argument("--topK", type=int, default=50, help="近邻交互的规模")
parser.add_argument("--path_topK_mode", type=str, default='eachPathTopK',choices=['eachPathTopK','allPathsTopK'],
                    help= '近邻选取的方式')
parser.add_argument("--NI_sim_mode", type=str, default='PasRec_2path',choices=['PasRec','PasRec_2path','IsRec','IsRec_best'],
                    help= '近邻交互时计算相似度的方式')
parser.add_argument("--new_HIN_paras", type=list, default=[None, None, 'Deep', None, 'Deep'], help="近邻交互时计算相似度的方式")


# --隐式交互
parser.add_argument("--if_implict", type=bool, default=True, help="是否隐式交互")
parser.add_argument("--mf_mode", type=str, default='node2vec',choices=['pmf','nmf','node2vec','BiNE','BPR'],help= '隐式交互embedding方式')
parser.add_argument("--implict_feat_dim", type=int, default=25, help="隐式交互隐向量维度")
parser.add_argument("--imp_fc_unit_nums", type=list, default=[100, 50], help="隐式交互MLP结构")
# parser.add_argument("--CF_self_1st_merge", type=bool, default=True, help="隐式表示是否先用MLP处理，还是元素乘") # TODO

parser.add_argument("--NI_handle_slt_apis_mode", type=str, default='attention',choices=['attention', 'full_concate', 'average', ''],
                    help= 'NI中处理多个已选服务的方式')
parser.add_argument("--simple_NI_slt_mode", type=str, default='att',choices=['att', 'ful', 'ave', ''],
                    help= 'CI中处理多个已选服务的方式')
# --显式交互
parser.add_argument("--if_explict", type=int, default=0, help="是否显式交互及其类型: 0没有 1是跟所有api的共现次数向量 2是跟最近邻mashup调用过的api的共现次数 3是最近邻mashup是否调用过该api，50D") # 不同数字代表几种不同模式
parser.add_argument("--exp_fc_unit_nums", type=list, default=[64, 16], help="显式交互MLP结构") # [1024, 256, 64, 16]

# --可组合信息
parser.add_argument("--if_correlation", type=bool, default=False, help="是否加入可组合信息")
parser.add_argument("--cor_fc_unit_nums", type=list, default=[128, 64, 16], help="可组合信息处理MLP结构")

# --最终预测
parser.add_argument("--predict_fc_unit_nums", type=list, default=[128, 64, 32], help="最终预测MLP结构")
parser.add_argument("--final_activation", type=str, default='softmax',choices=['softmax','sigmoid'],help= '最终激活函数')

# 3 model训练和评估
parser.add_argument("--train_mode", type=str, default='best_NDCG', choices=['best_NDCG','min_loss'],help="训练终止条件")
parser.add_argument("--train_new", type=bool, default=False,help="是否重新训练") # TODO
parser.add_argument("--pairwise", type=bool, default=False, help="优化是否pairwise")
parser.add_argument("--margin", type=float, default=0, help="hinge pairwise loss")
parser.add_argument("--embedding_train", type=bool, default=True, help="是否训练词embedding")
parser.add_argument("--embeddings_regularizer", type=float, default=0, help="词embedding L2")
parser.add_argument("--CI_learning_rate", type=float, default=0.0003, help="CI学习率")
parser.add_argument("--NI_learning_rate", type=float, default=0.0003, help="NI学习率")
parser.add_argument("--topMLP_learning_rate", type=float, default=0.0001, help="topMLP学习率")
parser.add_argument("--l2_reg", type=float, default=0, help="MLP L2")
parser.add_argument("--num_epochs", type=int, default=1, help="训练轮数")
parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
parser.add_argument("--test_batch_size", type=int, default=64, help="test batch_size")
parser.add_argument("--validation_split", type=float, default=0.2, help="训练集验证集比例")
parser.add_argument("--train_name", type=str, default=None)
parser.add_argument("--kcv", type=int, default=5, help="交叉验证折数")
parser.add_argument("--topKs", type=list, default=[k for k in range(5, 55, 5)], help="测试NDCG用")

# 5 其他模型参数
parser.add_argument("--sim_feature_size", type=int, default=8, help="DHSR model中相似度特征维度")
parser.add_argument("--mf_fc_unit_nums", type=list, default=[32, 16, 8], help="DHSR 内容部分MLP结构")
parser.add_argument("--mf_embedding_dim", type=int, default=50, help="DHSR MF特征维度")
parser.add_argument("--final_MLP_layers", type=list, default=[120, 50], help="DHSR最终整合MLP结构")
parser.add_argument("--DHSR_lr", type=float, default=0.003, help="lr")

parser.add_argument("--NCF_layers", type=list, default=[64, 32, 16, 8], help="NCF MLP结构")
parser.add_argument("--NCF_reg_layers", type=list, default=[0.01, 0.01, 0.01, 0.01], help="NCF MLP各层的L2正则")
parser.add_argument("--NCF_reg_mf", type=float, default=0.01, help="DHSR MF部分正则")
parser.add_argument("--NCF_lr", type=float, default=0.003, help="lr")

args = parser.parse_args()

# 更新设置其他参数
args.simple_CI_slt_mode = 'nul' if not args.CI_handle_slt_apis_mode else args.CI_handle_slt_apis_mode[:3]
args.simple_NI_slt_mode = 'nul' if not args.NI_handle_slt_apis_mode else args.NI_handle_slt_apis_mode[:3]

args.cur_data_dir = new_data_dir if args.which_data == 'new' else old_data_dir # 新旧数据集***
args.train_name = '_TRAIN:{}_need_slt_apis:{}_l2:{}'.format(args.train_mode, args.need_slt_apis,args.l2_reg)

# args.text_extracter_mode = 'bert' # 实验 bert
# args.batch_size = 8 # 一般64，但BERT要使用小batch size!
# args.test_batch_size = 16
# args.frozen_bert = True
data_repository.set_args(args)


# # 预处理, 初始化meta_data对象
# meta_data.initilize(args)
data_repository.set_md(args)


index = 0
for a_dataset in KCV_dataset_generater(args): # 划分数据集
    print('getting the {}th kcv...'.format(index))
    index += 1

start_index,end_index = 0,0
index = 0
for a_dataset in KCV_dataset_generater(args):
    print('kcv:{}'.format(index))
    if index < start_index:
        index += 1
        continue
    if index > end_index:
        break

    # dataset.set_cur_dataset(a_dataset) # 已划分的数据集对象
    index += 1

    # cotrain_CINI()
    # baselines(a_dataset)
    # bl_DHSR()
    # bl_DHSR_new(a_dataset)
    # text_tag()

    CI_NI_fineTuning()

    # NI_online() # 最新的模型
    # co_trainCINI()
    # test_PNCF_doubleTower_OR_DIN()

    # bl_IsRec()
    # bl_IsRec_best(a_dataset)
    # bl_PasRec(a_dataset)
    # deepFM()
    # newDeepFM() # 效果很差
    # test()
    # a_dataset.transfer() # 新数据对不加已选的模型的简化, 用在CI和NI的need_slt_apis = False中
    # test_simModes(a_dataset,old_new,if_few = if_few)
    # DINRec(a_dataset,old_new)



