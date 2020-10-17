# -*- coding:utf-8 -*-
import itertools
import os
import sys
import random
import pandas as pd

from core.data_repository import set_ds
from core.dataset import dataset
from core.process_raw_data import meta_data
from Helpers.util import list2dict


# 将meta_data.mashup_api_list划分成训练集和测试集
# 然后得到对应的dataset对象


def KCV_dataset_generater(args):
    if args.data_mode == 'newScene':
        dataset_generator = split_dataset_for_newScene_KCV(args)
    elif args.data_mode == 'oldScene':
        dataset_generator = split_dataset_for_oldScene_KCV(args)
    return dataset_generator


def split_dataset_for_oldScene_KCV(args):
    data_dir = args.cur_data_dir
    num_negatives = args.num_negatives
    test_candidates_nums = args.test_candidates_nums
    kcv = args.kcv

    name = 'oldScene_neg{}_testCandi{}'.format(num_negatives, test_candidates_nums)
    result_root = os.path.join(data_dir, 'split_data_oldScene', name)
    mashup_api_list = meta_data(args).mashup_api_list
    mashup_api_dict = list2dict(mashup_api_list)

    # 返回K个数据对象，每个可以使用相关属性
    if os.path.exists(dataset(args,result_root, name, kcv - 1).train_df_path):  # 已经划分过
        for i in range(kcv):
            print('has splited data in kcv mode before,read them!')
            data = dataset(args,result_root, name, i)
            data.initialize()  # 从文件中读取对象
            set_ds(data) # 设置唯一实例
            yield data
    else:  # 还未划分过
        mashup_ids = list(mashup_api_dict.keys())
        all_apis = set(meta_data(args).api_df.index.tolist())  # 所有api的id

        # 首先为每个mashup指定确定的正负例，候选api等
        # {mid:api_instances}
        mid2true_instances, mid2false_instances, mid2candidate_instances = {}, {}, {}
        for mashup_id, api_ids in mashup_api_dict.items():  # api_ids是set
            unobserved_apis_list = list(all_apis - api_ids)
            random.shuffle(unobserved_apis_list)

            mid2true_instances[mashup_id] = {}
            mid2false_instances[mashup_id] = {}
            mid2candidate_instances[mashup_id] = {}

            api_ids_list = list(api_ids)  # 已选择的apis，做正例
            mid2true_instances[mashup_id] = api_ids_list

            all_neg_num = min(meta_data(args).api_num, num_negatives * len(api_ids_list))
            mid2false_instances[mashup_id] = unobserved_apis_list[:all_neg_num]  # 负例

            if test_candidates_nums == 'all':  # 选取全部api做测试
                mid2candidate_instances[mashup_id] = list(all_apis)
            else:  # 选取部分作为测试，实际组件api和部分unobserved
                test_candidates_nums = int(test_candidates_nums)
                mid2candidate_instances[mashup_id] = api_ids_list + unobserved_apis_list[:test_candidates_nums]

        random.shuffle(mashup_ids)
        batch = len(mashup_ids) // kcv
        for i in range(kcv):  # 每个kcv
            start_index = i * batch
            batch_stopindex = len(mashup_ids) if i == kcv - 1 else (i + 1) * batch
            test_mashups = mashup_ids[start_index:batch_stopindex]
            train_mashups = mashup_ids[:start_index] + mashup_ids[batch_stopindex:-1]

            train_df = pd.DataFrame(columns=['mashup', 'api', 'label'])
            for mashup_id in train_mashups:
                for true_api_id in mid2true_instances[mashup_id]:
                    train_df.append({'mashup': mashup_id, 'api': true_api_id, 'label': 1},ignore_index=True)
                for false_api_id in mid2false_instances[mashup_id]:
                    train_df.append({'mashup': mashup_id, 'api': false_api_id, 'label': 0}, ignore_index=True)

            # test和train格式不同
            # test mashup和api的一行list是多个测试样本,而all_ground_api_ids,test_slt_ids的一行对应前者的一行
            test_df = pd.DataFrame(columns=['mashup', 'slt_apis', 'candidate_apis', 'all_ground_api_ids'])
            for mashup_id in test_mashups:
                test_df.append(
                    {'mashup': mashup_id, 'candidate_apis': mid2candidate_instances[mashup_id],
                     'all_ground_api_ids': mid2true_instances[mashup_id]}, ignore_index=True)

            data = dataset(args,result_root, name, i)
            data.initialize(train_df, test_df)
            set_ds(data)  # 设置唯一实例
            print('{}/{} dataset, build done!'.format(i, kcv))
            yield data


def split_dataset_for_newScene_KCV(args):
    """
    新场景划分数据
    :param data_dir: 要划分数据的路径
    :param num_negatives: 负采样比例
    :param slt_num: 指定的最大已选择服务的数目
    :param slt_combination_num: 真实组件服务中,只选取一部分组合作为已选服务,缓解数据不平衡问题: eg: C10/3 C50/3
    # :param train_positive_samples: 每个训练用的mashup,除了已选服务,剩下的保留多少个服务作为训练正例，防止组件太多的mashup所占的比例太大
    :param test_candidates_nums: 每个mashup要评价多少个待测负例item: 为all时全部评价
    :param kcv:
    :return: 某折的dataset对象
    """
    data_dir = args.cur_data_dir
    num_negatives = args.num_negatives
    slt_num = args.slt_item_num
    slt_combination_num = args.combination_num
    test_candidates_nums = args.test_candidates_nums
    kcv = args.kcv

    name = 'newScene_neg{}_sltNum{}_com{}_testCandi{}'.format(num_negatives, slt_num, slt_combination_num,
                                                       test_candidates_nums)
    result_root = os.path.join(data_dir, 'split_data_newScene', name)

    # 返回K个dataset对象，每个可以使用相关属性
    if os.path.exists(dataset(args,result_root, name, kcv - 1).train_df_path):  # 已经划分过
        for i in range(kcv):
            print('data has been splited in kcv mode before,read them!')
            data = dataset(args,result_root, name, i)
            data.initialize()  # 从文件中读取对象
            set_ds(data)  # 设置唯一实例
            yield data
    else:
        mashup_api_list = meta_data(args).mashup_api_list
        mashup_api_dict = list2dict(mashup_api_list)
        mashup_ids = meta_data(args).mashup_df.index.tolist()
        mashup_ids.remove(0) # 占位
        all_apis = set(meta_data(args).api_df.index.tolist())  # 所有api的id
        all_apis.remove(0)

        # 1.首先为每个mashup指定已选服务和对应的正负例(训练)/待测api(测试)
        # {mid:{slt_aid_list:api_instances}
        mid2true_instances, mid2false_instances, mid2candidate_instances = {}, {}, {}
        for mashup_id, api_ids in mashup_api_dict.items():  # api_ids是set
            unobserved_apis_list = list(all_apis - api_ids)
            random.shuffle(unobserved_apis_list)

            mid2true_instances[mashup_id] = {}
            mid2false_instances[mashup_id] = {}
            mid2candidate_instances[mashup_id] = {}

            api_ids_list = list(api_ids)
            max_slt_num = min(slt_num, len(api_ids_list) - 1)  # eg:最大需要三个已选服务，但是只有2个services构成
            for act_slt_num in range(max_slt_num):  # 选择1个时，两个时...
                act_slt_num += 1
                combinations = list(itertools.combinations(api_ids_list, act_slt_num))
                if slt_combination_num != 'all':  # 只选取一部分组合,缓解数据不平衡问题
                    slt_combination_num = min(len(combinations), slt_combination_num)
                    combinations = combinations[:slt_combination_num]

                for slt_api_ids in combinations:  # 随机组合已选择的api，扩大数据量 # 组合产生,当做已选中的apis
                    train_api_ids = list(api_ids - set(slt_api_ids))  # masked observed interaction 用于训练或测试的

                    # if train_positive_samples != 'all':  # 选择一部分正例 做训练或测试
                    #     train_positive_samples_num = min(len(train_api_ids), train_positive_samples) # 最多50个，一般没有那么多
                    #     train_api_ids = train_api_ids[:train_positive_samples_num]

                    mid2true_instances[mashup_id][slt_api_ids] = train_api_ids  # 训练用正例 slt_api_ids是tuple

                    num_negative_instances = min(num_negatives * len(train_api_ids), len(unobserved_apis_list))
                    mid2false_instances[mashup_id][slt_api_ids] = unobserved_apis_list[
                                                                  :num_negative_instances]  # 随机选择的负例

                    if test_candidates_nums == 'all':  # 待预测
                        test_candidates_list = list(all_apis - set(slt_api_ids))
                    else:
                        test_candidates_nums = int(test_candidates_nums)
                        test_candidates_list = unobserved_apis_list[:test_candidates_nums] + train_api_ids
                    mid2candidate_instances[mashup_id][slt_api_ids] = test_candidates_list

        random.shuffle(mashup_ids)
        batch = len(mashup_ids) // kcv
        # 2.然后，根据上面的结果划分为各个KCV，训练和测试
        for i in range(kcv):  # 每个kcv
            start_index = i * batch
            batch_stopindex = len(mashup_ids) if i == kcv - 1 else (i + 1) * batch
            test_mashups = mashup_ids[start_index:batch_stopindex]
            train_mashups = mashup_ids[:start_index] + mashup_ids[batch_stopindex:-1]
            print(train_mashups)
            print(test_mashups)

            train_df = pd.DataFrame(columns=['mashup', 'slt_apis', 'api', 'label'])
            for mashup_id in train_mashups:
                for slt_api_ids, true_api_instances in mid2true_instances[mashup_id].items():
                    for true_api_id in true_api_instances:
                        train_df = train_df.append({'mashup': mashup_id, 'slt_apis': slt_api_ids, 'api': true_api_id, 'label': 1},
                                        ignore_index=True)

                for slt_api_ids, false_api_instances in mid2false_instances[mashup_id].items():
                    for false_api_id in false_api_instances:
                        train_df = train_df.append({'mashup': mashup_id, 'slt_apis': slt_api_ids, 'api': false_api_id, 'label': 0},
                                        ignore_index=True)

            # test和train格式不同: train是一行一个样本; test的待测api太多，节省空间，一行:
            # 一个mashup，多个待测api,一份all_ground_api_ids,一份test_slt_ids
            test_df = pd.DataFrame(columns=['mashup', 'slt_apis', 'candidate_apis', 'all_ground_api_ids'])
            for mashup_id in test_mashups:
                for slt_api_ids, candidate_api_instances in mid2candidate_instances[mashup_id].items():
                    test_df = test_df.append(
                        {'mashup': mashup_id, 'slt_apis': slt_api_ids, 'candidate_apis': candidate_api_instances,
                         'all_ground_api_ids': mid2true_instances[mashup_id][slt_api_ids]}, ignore_index=True)

            # 3.根据训练集和测试集的划分，初始化dataset对象!!!
            data = dataset(args,result_root, name, i)
            data.initialize(train_df, test_df)
            set_ds(data)  # 设置唯一实例
            print('{}/{} dataset, build done!'.format(i, kcv))
            yield data

    print('you have splited and saved them!')
