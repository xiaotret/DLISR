import os

# 项目的根目录
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# path_root = path_root.replace('\\', '/')

# 原始数据路径
new_data_dir = os.path.join(path_root, 'data','new')
old_data_dir = os.path.join(path_root, 'data','old')
glove_embedding_path = os.path.join(path_root, 'pre_trained_embeddings')
google_embedding_path = os.path.join(path_root, 'pre_trained_embeddings') #

# 评估结果路径: 在根目录下，唯一
result_root = os.path.join(path_root, 'result')
if not os.path.exists(result_root):
    os.makedirs(result_root)
evaluate_path = os.path.join(result_root, 'evaluate.csv')
loss_path = os.path.join(result_root,'loss_acc.csv')
time_path = os.path.join(result_root, 'time.csv')

# 中间结果
features_result_path = os.path.join(result_root, 'feature')
history_result_path = os.path.join(result_root, 'history')