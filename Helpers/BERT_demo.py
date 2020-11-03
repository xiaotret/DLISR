from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification

# 特征提取
from transformers import DistilBertTokenizer, TFDistilBertModel
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='tf')
# {'input_ids': <tf.Tensor: shape=(1, 12), dtype=int32, numpy= array([[ 101, 5672, 2033, 2011, 2151, 3793, 2017, 1005, 1040, 2066, 1012, 102]])>,
# 'attention_mask': <tf.Tensor: shape=(1, 12), dtype=int32, numpy=array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])>}
output = model(encoded_input) # 得到特征向量
# (<tf.Tensor: shape=(1, 12, 768), dtype=float32, numpy= ..., dtype=float32)>,)



# 分类任务
def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir is "neg" else 1)

    return texts, labels


# train_texts, train_labels = read_imdb_split('aclImdb/train')
# test_texts, test_labels = read_imdb_split('aclImdb/test')
#
# train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
# tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
# train_encodings = tokenizer(train_texts, truncation=True, padding=True)
# val_encodings = tokenizer(val_texts, truncation=True, padding=True)
# test_encodings = tokenizer(test_texts, truncation=True, padding=True)
#
# train_dataset = tf.data.Dataset.from_tensor_slices((
#     dict(train_encodings),
#     train_labels
# ))
# val_dataset = tf.data.Dataset.from_tensor_slices((
#     dict(val_encodings),
#     val_labels
# ))
# test_dataset = tf.data.Dataset.from_tensor_slices((
#     dict(test_encodings),
#     test_labels
# ))

# model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased') # keras model
# distilbert_model = model.get_layer('distilbert')
# model.summary()

# optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
# model.compile(optimizer=optimizer, loss=model.compute_loss) # can also use any keras loss fn
# model.fit(train_dataset.shuffle(1000).batch(16), epochs=3, batch_size=16)