import sys
import os
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
import matplotlib.pyplot as plt

from core.evalute import evalute_by_epoch, summary
from core import data_repository
from core.path_config import evaluate_path, loss_path
from recommend_models.CI_Model import CI_Model


def train_best_NDCG_model(recommend_model, model, train_data, test_data, true_candidates_dict=None,
                          CI_start_test_epoch=0, earlyStop_epochs=5):
    """
    训练多个epoch，每个之后均测试，选择并返回NDCG等最终指标最优的模型
    :param recommend_model:  整体的推荐模型
    :param model:  model_core
    :param train_data:
    :param test_data:
    :param start_epoch: 之前该模型已经训练过多个epoch，在这个基础上接着训练
    :param true_candidates_dict:
    :return:
    """
    print('training_save_best_NDCG_model...')
    epoch_evaluate_results = []

    # 模型
    train_model = recommend_model.get_pairwise_model() if data_repository.get_args().pairwise else model

    # 数据
    train_instances_dict = recommend_model.get_instances(train_data,
                                                         pairwise_train_phase_flag=data_repository.get_args().pairwise)
    train_labels = train_data.get('label')
    if data_repository.get_args().final_activation == 'softmax':  # 针对softmax变换labels
        train_labels = utils.to_categorical(train_labels, num_classes=2)

    best_epoch, best_NDCG_5 = 0, 0
    for epoch in range(data_repository.get_args().num_epochs):
        if epoch == 0:  # 首次训练要编译
            # loss_ = lambda y_true, y_pred: y_pred if data_repository.get_args().pairwise else 'binary_crossentropy'
            # train_model.compile(optimizer=recommend_model.optimizer, loss=loss_,metrics=['accuracy'])
            train_model.compile(optimizer=recommend_model.optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            print('whole_model compile,done!')
        print('Epoch {}'.format(epoch))

        hist = train_model.fit(train_instances_dict, np.array(train_labels),
                               batch_size=data_repository.get_args().batch_size, epochs=1, verbose=1, shuffle=True,
                               validation_split=data_repository.get_args().validation_split)
        print('Epoch {}, train done!'.format(epoch))

        # 记录：数据集情况，模型架构，训练设置
        record_name = recommend_model.get_name() + data_repository.get_args().train_name if epoch == 0 else ''  # 记录在测试集的效果，写入evalute.csv
        save_loss_acc(hist, record_name, epoch=epoch)  # 每个epoch记录

        # CI的前3轮效果差，一般不用测，提高速度
        first_test_epoch = CI_start_test_epoch  if isinstance(recommend_model, CI_Model) else 0
        if epoch < first_test_epoch:
            epoch_evaluate_results.append(None)
            continue

        # epoch测试
        epoch_evaluate_result = evalute_by_epoch(recommend_model, model, record_name, test_data,
                                                 record_time=True if epoch == 0 else False,
                                                 true_candidates_dict=true_candidates_dict)
        epoch_evaluate_results.append(epoch_evaluate_result)

        # 优于目前的best_NDCG_5才存储模型参数 TODO
        if epoch_evaluate_result[0][3] >= best_NDCG_5:
            best_NDCG_5 = epoch_evaluate_result[0][3]
            best_epoch = epoch
            model.save_weights(data_repository.get_ds().new_model_para_path.format(recommend_model.model_dir,epoch))
        else:
            if epoch - best_epoch >= earlyStop_epochs:  # 大于若干个epoch，效果没有提升，即时终止
                break

    # 记录最优epoch和最优NDCG@5
    with open(data_repository.get_ds().new_best_epoch_path.format(recommend_model.model_dir), 'w') as f:
        f.write(str(best_epoch))
    with open(data_repository.get_ds().new_best_NDCG_path.format(recommend_model.model_dir), 'w') as f:
        f.write(str(best_NDCG_5))
    print('best epoch:{},best NDCG@5:{}'.format(best_epoch, best_NDCG_5))

    # 记录最优指标
    csv_table_name = 'best_indicaters\n'
    summary(evaluate_path, csv_table_name, epoch_evaluate_results[best_epoch], data_repository.get_args().topKs)

    # 看word embedding矩阵是否发生改变，尤其是padding的0
    # print('some embedding parameters after {} epoch:'.format(epoch))
    # print (recommend_model.embedding_layer.get_weights ()[0][:2])

    # 把记录的非最优的epoch模型参数都删除
    try:
        for i in range(data_repository.get_args().num_epochs):
            temp_path = data_repository.get_ds().new_model_para_path.format(recommend_model.model_dir, i)
            if i != best_epoch and os.path.exists(temp_path):
                os.remove(temp_path)
        model.load_weights(data_repository.get_ds().new_model_para_path.format(recommend_model.model_dir, best_epoch))
    finally:
        return model


def save_loss_acc(train_log, model_name, epoch=0, if_multi_epoch=False):
    # if_multi_epoch：每次存一个epoch
    # 每个epoch存储loss,val_loss,acc,val_acc
    if not if_multi_epoch:
        with open(loss_path, 'a+') as f:
            if epoch == 0:  # 第一个epoch记录模型名
                f.write(model_name + '\n')
                if data_repository.get_args().validation_split == 0:
                    f.write('epoch,loss,acc\n')
                else:
                    f.write('epoch,loss,val_loss,acc,val_acc\n')
            if data_repository.get_args().validation_split == 0:
                f.write('{},{},{}\n'.format(epoch, train_log.history["loss"][0], train_log.history["acc"][0]))
            else:
                f.write('{},{},{},{},{}\n'.format(epoch, train_log.history["loss"][0], train_log.history["val_loss"][0],
                                                  train_log.history["accuracy"][0], train_log.history["val_accuracy"][0]))
    else:
        with open(data_repository.get_args().loss_path, 'a+') as f:
            f.write(model_name + 'EarlyStop' + '\n')
            if data_repository.get_args().validation_split == 0:
                f.write('epoch,loss,acc\n')
            else:
                f.write('epoch,loss,val_loss,acc,val_acc\n')
            epoch_num = len(train_log.history["loss"])
            for i in range(epoch_num):
                if data_repository.get_args().validation_split == 0:
                    f.write('{},{},{}\n'.format(i, train_log.history["loss"][i], train_log.history["acc"][i]))
                else:
                    f.write('{},{},{},{},{}\n'.format(i, train_log.history["loss"][i], train_log.history["val_loss"][i],
                                                      train_log.history["acc"][i], train_log.history["val_acc"][i]))


def train_monitoring_loss_acc_model(recommend_model, model, train_data):
    """
    绘制loss_acc曲线, 观察过拟合欠拟合
    """
    train_labels = train_data[-1]
    train_instances_tuple = recommend_model.get_instances(*train_data[:-1])
    model.compile(optimizer=Adam(lr=data_repository.get_args().learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    hist = model.fit([*train_instances_tuple], np.array(train_labels), batch_size=data_repository.get_args().small_batch_size,
                     epochs=data_repository.get_args().num_epochs,
                     verbose=1, shuffle=True, validation_split=0.1)  # 可以观察过拟合欠拟合
    plot_loss_acc(hist, recommend_model.get_simple_name())
    return model


def plot_loss_acc(train_log, model_name):
    # 传入log对象，绘制曲线
    epochs = data_repository.get_args().num_epochs
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), train_log.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), train_log.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), train_log.history["acc"], label="train_acc")
    plt.plot(np.arange(0, epochs), train_log.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on the whole_model")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper right")
    plt.savefig("Loss_Accuracy_{}.jpg".format(model_name))


def train_early_stop(recommend_model, model, train_data, test_data):
    """
    训练时按照验证集的loss，early stopping得到最优的模型；最后基于该模型测试
    :return:
    """
    if_Train = True if data_repository.get_args().pairwise else False
    train_labels = train_data[-1]
    train_instances_tuple = recommend_model.get_instances(*train_data[:-1], pairwise_train_phase_flag=if_Train)

    train_model = recommend_model.get_pairwise_model() if data_repository.get_args().pairwise else model
    if data_repository.get_args().pairwise:
        train_model.compile(optimizer=recommend_model.optimizer, loss=lambda y_true, y_pred: y_pred,
                            metrics=['accuracy'])
    else:
        train_model.compile(optimizer=recommend_model.optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2, mode='min')
    hist = train_model.fit([*train_instances_tuple], train_labels, epochs=data_repository.get_args().num_epochs,
                           batch_size=data_repository.get_args().small_batch_size, callbacks=[early_stopping],
                           validation_split=data_repository.get_args().validation_split, shuffle=True)  #
    model.save_weights(
        data_repository.get_ds().new_model_para_path.format(recommend_model.model_dir, 'min_loss'))  # !!! 改正

    model_name = recommend_model.get_simple_name() + recommend_model.get_name() + '_min_loss'
    save_loss_acc(hist, model_name, if_multi_epoch=True)

    epoch_evaluate_result = evalute_by_epoch(recommend_model, model, model_name, test_data)
    return model


def train_model(recommend_model, model, train_data, test_data, train_mode, retrain=True,true_candidates_dict=None):
    """
    各种模型(完全冷启动和部分冷启动，完整和部分的)都可以通用
    :param recommend_model:
    :param model:
    :param train_data: 与参数对应，是否加入slt_api_ids
    :param test_data:
    :param train_mode： 'best_NDCG' or 'min_loss'
    :param retrain： 是否重新训练模型
    :return:
    """
    # 模型相关的东西都放在该数据下的文件夹下,不同模型不同文件夹！！！

    model_dir = recommend_model.model_dir
    if not os.path.exists(model_dir):
        print('makedirs for:', model_dir)
        os.makedirs(model_dir)

    if os.path.exists(data_repository.get_ds().new_best_epoch_path.format(model_dir)) and not retrain:  # 加载求过的结果
        print('preTrained whole_model, exists!')
        return load_pretrained_model(recommend_model, model)
    else:
        if train_mode == 'best_NDCG':
            model = train_best_NDCG_model(recommend_model, model, train_data, test_data,
                                          true_candidates_dict=true_candidates_dict)
        elif train_mode == 'min_loss':
            model = train_early_stop(recommend_model, model, train_data, test_data)
        elif train_mode == 'monitor loss&acc':
            train_monitoring_loss_acc_model(recommend_model, model, train_data, test_data)
        else:
            print('wrong train_mode:')
            print(train_mode)
        return model


def load_pretrained_model(recommend_model, model):
    """
    只需要载入并返回训练好的模型即可
    :param recommend_model:
    :param para_mode:
    :return:
    """
    with open(data_repository.get_ds().new_best_epoch_path.format(recommend_model.model_dir), 'r') as f:
        best_epoch = int(f.readline())
    para_path = data_repository.get_ds().new_model_para_path.format(recommend_model.model_dir, best_epoch)
    model.load_weights(para_path)
    print('load whole_model:{},done!'.format(recommend_model.simple_name))
    return model