import numpy as np
from tensorflow.keras.utils.data_utils import Sequence
from tensorflow.keras.utils.np_utils import to_categorical
from core import data_repository

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, ids_dict, train_test,batch_size=32, shuffle=True):
        'Initialization'
        self.batch_num = self.__len__()
        self.batch_size = batch_size
        self.ids_dict = ids_dict
        self.m_ids = ids_dict.get('mashup')
        self.a_ids = ids_dict.get('api')
        self.slt_apis = ids_dict.get('slt_apis')
        self.labels = ids_dict.get('label')
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.m_ids) / self.batch_size))+1

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        end_index = min(self.batch_num,(index+1)*self.batch_size)
        indexes = self.indexes[index*self.batch_size:end_index]

        m_ids = [self.m_ids[k] for k in indexes]
        a_ids = [self.a_ids[k] for k in indexes]
        slt_apis = [self.slt_apis[k] for k in indexes] if self.slt_apis else None
        labels = [self.labels[k] for k in indexes] if self.labels else None

        # Generate data
        X, y = self.__data_generation(m_ids,a_ids,slt_apis,labels)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.m_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, m_ids,a_ids,slt_apis,labels):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, y