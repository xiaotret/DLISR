import csv
import heapq
import os
import sys
import time

from core import data_repository
from Helpers.evaluator import evaluate

import numpy as np
from prettytable import PrettyTable


import pickle

from core.path_config import time_path, evaluate_path


def evalute_by_epoch(recommend_model, model, model_name, test_data, show_cases=0, record_time=False,
                     true_candidates_dict=None, if_save_recommend_result=False, evaluate_by_slt_apiNum=False):
    """
    对训练好的模型，进行测试：可以适用于完全冷启动和部分冷启动情景；
    :param show_cases: 显示几个推荐结果
    :param record_time: 记录测试集的处理时间
    :param true_candidates_dict: 重排序：是否使用IsRec等算法的处理方式：近邻没有调用过的服务评分设置为0 mashup id -> api ids list
    :param if_save_recommend_result: 是否存储每个样例的推荐结果，用于case分析
    :param evaluate_by_slt_apiNum: 是否按照已选服务的数目将测试集分开评价
    :return:
    """

    # 某个已选择api数目下的测试集样本
    test_mashup_id_list = test_data.get('mashup')
    test_api_id_list = test_data.get('api')
    grounds = test_data.get('all_ground_api_ids')
    test_slt_ids = test_data.get('slt_apis')

    csv_table_name = model_name + '\n'
    test_instance_num = len(test_mashup_id_list)

    # 获取所有的预测结果
    def get_predictions():
        predictions = []  # 测试样本一次的预测结果
        for i in range(test_instance_num):
            candidate_ids = test_api_id_list[i]
            test_batch_size = data_repository.get_args().test_batch_size
            prediction = []

            # test api 太多，手动分batch预测
            test_api_num = len(candidate_ids)
            batch_num = test_api_num // test_batch_size
            remainder = test_api_num % test_batch_size
            if remainder != 0:
                batch_num += 1
            start_time = time.time()
            for j in range(batch_num):  # 每个batch
                start_index = j * test_batch_size
                stop_index = test_api_num if (remainder != 0 and j == batch_num - 1) else (j + 1) * test_batch_size
                batch_api_ids = candidate_ids[start_index:stop_index]

                batch_instances_dict = { 'mashup':test_mashup_id_list[i][start_index:stop_index],'api':batch_api_ids}
                if data_repository.get_args().data_mode == 'newScene' and data_repository.get_args().need_slt_apis: # TODO
                    _slt_ids = []
                    _slt_ids.append(test_slt_ids[i])  # 同一行的同个mashup对各个api的评分中，已选择的apis一样
                    batch_instances_dict['slt_apis'] = _slt_ids * (stop_index - start_index)

                batch_instances_dict = recommend_model.get_instances(batch_instances_dict)
                batch_prediction = model.predict(batch_instances_dict)

                if len(batch_prediction.shape) == 2:
                    batch_prediction = batch_prediction[:, 1]  # 1:[0,1]
                batch_prediction = list(batch_prediction)
                prediction += batch_prediction  # 一个mashup对所有候选的评分
            predictions.append(list(prediction))

            end_time = time.time()
            if record_time:
                with open(time_path, 'a+') as f1:
                    if i ==0:
                        f1.write(recommend_model.get_simple_name())
                        f1.write('\n')
                    f1.write('num of instances,{},cost time,{}\n'.format(test_api_num,end_time - start_time))

            # 展示几个mashup的推荐结果
            def show_prediction_res(i):
                print('for mashup {}:'.format(test_mashup_id_list[i][0]))
                if data_repository.get_args().need_slt_apis:
                    print('slt_ids:', test_slt_ids[i])
                sorted_pre2id = sorted(zip(prediction, test_api_id_list[i]))
                sorted_pres, sorted_ids = zip(*sorted_pre2id)
                print('candidate api ids', sorted_ids)
                print('predictions', sorted_pres)
                print('grounds', grounds[i])

            if i < show_cases:
                show_prediction_res(i)
            if i % 100 == 0:
                print('has test {}/{} mashup instances'.format(i, test_instance_num))
        print('test,done!')
        return predictions

    predictions = get_predictions()

    # 使用IsRec_best的策略处理一下待测服务的评分：没有被近邻mashup调用过的服务，评分直接设置为0
    if true_candidates_dict is not None:
        for i in range(len(predictions)):  # 每个mashup index
            true_candidates_list = true_candidates_dict[test_mashup_id_list[i][0]]
            assert len(test_api_id_list[i]) == len(predictions[i])
            _num = len(test_api_id_list[i])
            for j in range(_num):
                if test_api_id_list[i][j] not in true_candidates_list:  # 如果一个待测api没有被近邻mashup调用过，评分为0
                    predictions[i][j] = 0

    # 根据实例的已选择数目分别测试
    if evaluate_by_slt_apiNum and data_repository.get_args().data_mode == 'newScene':
        def _filter(slt_apiNum):
            test_api_id_list_, predictions_, grounds_ = [], [], []
            for i in range(test_instance_num):
                if len(test_slt_ids[i]) == slt_apiNum:
                    test_api_id_list_.append(test_api_id_list[i])
                    predictions_.append(predictions[i])
                    grounds_.append(grounds[i])
            return test_api_id_list_, predictions_, grounds_
        for slt_apiNum in range(3):
            test_api_id_list_, predictions_, grounds_ = _filter(slt_apiNum + 1)
            evaluate_result = evalute(test_api_id_list_, predictions_, grounds_, data_repository.get_args().topKs)
            summary(evaluate_path, str(slt_apiNum + 1) + '_' + csv_table_name, evaluate_result,
                    data_repository.get_args().topKs)  #

    if if_save_recommend_result and data_repository.get_args().data_mode == 'newScene':
        recommend_result_path = os.path.join(recommend_model.model_dir, 'recommend_result.csv')
        evaluate_result = evalute(test_api_id_list, predictions, grounds, data_repository.get_args().topKs, test_mashup_id_list,
                                  test_slt_ids, recommend_result_path)  # 评价并记录结果
        summary(evaluate_path, csv_table_name, evaluate_result, data_repository.get_args().topKs)  #
    else:
        evaluate_result = evalute(test_api_id_list, predictions, grounds, data_repository.get_args().topKs)
        summary(evaluate_path, csv_table_name, evaluate_result, data_repository.get_args().topKs)  #

    return evaluate_result  # topKs*5个指标


# 用于读取存储的预测结果，再跟pop值结合重新评价
def add_pop_predictions(recommend_model, csv_table_name, epoch, pop_mode='sigmoid', a_pop_ratio=0.0):
    test_mashup_id_list, test_api_id_list, predictions = None, None, None
    with open(os.path.join(data_repository.get_args().data_dir, 'model_predictions_{}.dat'.format(epoch)), 'rb') as f:
        test_mashup_id_list, test_api_id_list, predictions = pickle.load(f)

    api_id2covec, api_id2pop = recommend_model.pd.get_api_co_vecs(pop_mode=pop_mode)

    # 乘积
    predictions_pop = []
    for m_index in range(len(predictions)):
        a_mashup_predictions = predictions[m_index]
        temp_preditions = []
        for a_index in range(len(a_mashup_predictions)):
            a_prediction = a_mashup_predictions[a_index]
            api_id = test_api_id_list[m_index][a_index]
            temp_preditions.append(api_id2pop[api_id] * a_prediction)
        predictions_pop.append(temp_preditions)
    evaluate_result_linear_sum = evalute(test_api_id_list, predictions_pop, data_repository.get_args().grounds,
                                         data_repository.get_args().topKs)  # 评价
    summary(evaluate_path, pop_mode + '_pop_prod\n' + csv_table_name, evaluate_result_linear_sum,
            data_repository.get_args().topKs)

    # 线性加权求和
    pop_ratios = [0.2 + 0.2 * i for i in range(5)]
    for pop_ratio in pop_ratios:
        predictions_pop_linear = []
        for m_index in range(len(predictions)):
            a_mashup_predictions = predictions[m_index]
            temp_preditions = []
            for a_index in range(len(a_mashup_predictions)):
                a_prediction = a_mashup_predictions[a_index]
                api_id = test_api_id_list[m_index][a_index]
                temp_preditions.append((1 - pop_ratio) * a_prediction + pop_ratio * api_id2pop[api_id])
            predictions_pop_linear.append(temp_preditions)

        evaluate_result_linear_sum = evalute(test_api_id_list, predictions_pop_linear, data_repository.get_args().grounds,
                                             data_repository.get_args().topKs)  # 评价
        summary(evaluate_path, pop_mode + '_pop_{}\n'.format(pop_ratio) + csv_table_name,
                evaluate_result_linear_sum, data_repository.get_args().topKs)

    predictions_pop_last = []
    for m_index in range(len(predictions)):
        # 首先根据score选出候选
        score_mapping = [pair for pair in zip(test_api_id_list[m_index], predictions[m_index])]
        max_k_pairs = heapq.nlargest(100, score_mapping, key=lambda x: x[1])  # 根据score选取top100*
        max_k_candidates, _ = zip(*max_k_pairs)
        # 然后仅根据pop rank
        temp_preditions = [api_id2pop[api_id] if api_id in max_k_candidates else -1 for api_id in
                           test_api_id_list[m_index]]
        predictions_pop_last.append(temp_preditions)

    evaluate_result_linear_sum = evalute(test_api_id_list, predictions_pop_last, data_repository.get_args().grounds,
                                         data_repository.get_args().topKs)  # 评价
    summary(evaluate_path, pop_mode + '_pop_last\n' + csv_table_name, evaluate_result_linear_sum,
            data_repository.get_args().topKs)


def evalute(candidate_api_ids_list, predictions, grounds, topKs, test_mashup_id_list=None, test_slt_ids=None,
            recommend_result_path=None):
    """
    :param candidate_api_ids_list: 用于测试的api id [[,]...] 2d
    :param predictions: 对应的该对的预测评分 2d
    :param grounds: 实际的调用api id 2d
    :param topKs:  哪些topK
    :return:
    """
    max_k = topKs[-1]
    instance_num = len(candidate_api_ids_list)
    result = np.zeros((instance_num, len(topKs), 5))
    recommend_list = []
    for index in range(instance_num):  # 单个mashup评价
        score_mapping = list(zip(candidate_api_ids_list[index], predictions[index]))
        max_k_pairs = heapq.nlargest(max_k, score_mapping, key=lambda x: x[1])  # 根据score选取top50
        max_k_candidates, _ = zip(*max_k_pairs)
        recommend_list.append(max_k_candidates)
        for k_idx, k in enumerate(topKs):  # 某个topK
            result[index, k_idx, :] = evaluate(max_k_candidates, grounds[index], k)  # 评价得到五个指标，K对NDCG等有用

    def list2str(list_):
        return ' '.join([str(id) for id in list_])

    # 存储推荐结果，写进csv文件
    if recommend_result_path and not os.path.exists(recommend_result_path):
        with open(recommend_result_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            header = ['mashup_id', 'slt_api_ids', 'recommend_list', 'grounds', 'hit_apis']
            writer.writerow(header)
            for index in range(instance_num):
                # print(test_mashup_id_list[index][0],test_slt_ids[index],recommend_list[index],grounds[index])
                hit_apis = set(recommend_list[index]).intersection(set(grounds[index]))
                writer.writerow(
                    [str(test_mashup_id_list[index][0]), list2str(test_slt_ids[index]), list2str(recommend_list[index]),
                     list2str(grounds[index]), list2str(hit_apis)])
            print('writerow,', index)
    return np.average(result, axis=0)


def analyze_result(recommend_model, topKs):
    """
    读取recommend_result_path中的评价结果，再使用其他指标(pop和冗余度)进行评价
    :param recommend_model:
    :param recommend_result_path:
    :param topKs:
    :return:
    """
    recommend_result_path = os.path.join(recommend_model.model_dir, 'recommend_result_new.csv')
    mashup_ids, slt_api_ids, recommend_lists, grounds = [], [], [], []

    def str2list(str_):
        list_ = str_.split(' ')
        return [int(id) for id in list_]

    with open(recommend_result_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mashup_ids.append(str2list(row['mashup_id']))
            slt_api_ids.append(str2list(row['slt_api_ids']))
            recommend_lists.append(str2list(row['recommend_list']))
            grounds.append(str2list(row['grounds']))
    instance_num = len(mashup_ids)
    api_id2info = meta_data.pd.get_mashup_api_id2info('api')
    _, api_id2pop = meta_data.pd.get_api_co_vecs(pop_mode='')
    api_categories = [get_mashup_api_allCategories('api', api_id2info, api_id, data_repository.get_args().Category_type) for api_id
                      in range(meta_data.api_num)]

    def evaluate_others(recommend_list):
        size = len(recommend_list)
        pop = sum([api_id2pop[api_id] for api_id in recommend_list]) / size
        union_tags = set()
        tag_sum_num = 0
        for api_id in recommend_list:
            tags = api_categories[api_id]
            union_tags = union_tags.union(set(tags))
            tag_sum_num += len(tags)
        redundance = 1 - len(union_tags) / tag_sum_num
        return np.array([pop, redundance])  # pop和冗余度

    def analyze():
        indicators_name = ['pop', 'redundancy']
        indicators = np.zeros((instance_num, len(topKs), len(indicators_name)))  # pop redundancy 看指标有哪些
        for index in range(instance_num):  # 单个mashup评价
            for k_idx, k in enumerate(topKs):  # 某个topK
                indicators[index, k_idx, :] = evaluate_others(recommend_lists[index][:k])  # 评价得到五个指标，K对NDCG等有用
        return np.average(indicators, axis=0)

    indicators = analyze()
    recommend_result_path = os.path.join(recommend_model.model_dir, 'recommend_other_indicators.csv')
    summary_others(recommend_result_path, recommend_model.simple_name, indicators, topKs)


def summary_others(evaluate_path, csv_table_name, evaluate_result, topKs, use_table=True, stream=sys.stdout):
    assert len(topKs) == len(evaluate_result)
    # console 打印 结果
    table = PrettyTable("TopK pop redundancy".split())

    for k_idx, topK in enumerate(topKs):
        table.add_row((topK, *("{:.4f}".format(val) for val in evaluate_result[k_idx])))
    stream.write(str(table))
    stream.write("\n")
    # csv 中保存结果
    csv_table = csv_table_name + "TopK,pop,redundancy\n"
    for k_idx, topK in enumerate(topKs):
        csv_table += "{:.4f},{:.4f},{:.4f}\n".format(topK, *evaluate_result[k_idx])
    with open(evaluate_path, 'a+') as f1:
        f1.write(csv_table)
    return 0


def summary(evaluate_path, csv_table_name, evaluate_result, topKs, use_table=True, stream=sys.stdout):
    assert len(topKs) == len(evaluate_result)
    # console 打印 结果
    table = PrettyTable("TopK Precision Recall F1 NDCG MAP".split())

    for k_idx, topK in enumerate(topKs):
        table.add_row((topK, *("{:.4f}".format(val) for val in evaluate_result[k_idx])))
    stream.write(str(table))
    stream.write("\n")
    # csv 中保存结果
    csv_table = csv_table_name + "TopK,Precision,Recall,F1,NDCG,MAP\n"
    for k_idx, topK in enumerate(topKs):
        csv_table += "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(topK, *evaluate_result[k_idx])
    with open(evaluate_path, 'a+') as f1:
        f1.write(csv_table)
    return 0
