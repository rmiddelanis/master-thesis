import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataCleaner:
    def __init__(self, config):
        self.data_file_name = config['data_file_name']
        self.max_data = config['max_data']
        self.target_variables = config['target_variables']
        self.sample_frequency = config['sample_frequency']
        self.test_size = config['test_size']
        self.sampling_size = config['sampling_size']
        self.seed = config['seed']
        self.data = Data()

    def load(self):
        raw_data = pickle.load(open(self.data_file_name, 'rb'))
        self._clean_data_(raw_data)
        self._set_labels_()
        self._split_data_()
        return self.data

    def _clean_data_(self, data):
        data = data.loc[data.index.values % self.sample_frequency == 0, :]
        if 't[s]' in data.columns:
            data = data.drop(['t[s]'], axis=1)
        if 'data_id' in data.columns:
            data = data.drop(['data_id'], axis=1)
        data = data.reset_index(drop=True)
        if self.max_data != -1:
            data = data.head(n=self.max_data)
        self.data.cleaned = data

    def _set_labels_(self):
        labels = self.data.cleaned.columns
        labels = labels.tolist()
        self.data.labels = labels
        self.data.is_target_label = np.zeros(len(labels), dtype=bool)

    # Note: Pot. implement weights for train-test split to account for different lengths of sequences (in data_id)
    def _split_data_(self):
        train_data = self.data.cleaned.iloc[:int(len(self.data.cleaned)*(1-self.test_size)), :]
        test_data = self.data.cleaned.iloc[int(len(self.data.cleaned)*(1-self.test_size)):, :]
        self.data.sampled = self.data.cleaned.loc[self.data.cleaned.index.values % self.sampling_size == 0,:]
        self.data.scaler.fit(pd.concat([train_data, test_data]))
        train_data = train_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)
        test_data_scaled = self.data.scaler.transform(test_data)
        train_data_scaled = self.data.scaler.transform(train_data)
        self.data.x_test, self.data.y_test = self._target_feature_split_(test_data_scaled)
        self.data.x_train, self.data.y_train = self._target_feature_split_(train_data_scaled)

    def _target_feature_split_(self, input_data):
        N_columns = input_data.shape[1]
        for idx, label in enumerate(self.data.labels):
            if label in self.target_variables:
                self.data.is_target_label[idx] = True
        features = input_data[:, ~self.data.is_target_label]
        targets = input_data[:, self.data.is_target_label]
        return features, targets


class Data:
    def __init__(self):
        self.cleaned = None
        self.x_train = None
        self.x_valid = None
        self.x_test = None
        self.y_train = None
        self.y_valid = None
        self.y_test = None
        self.train_split_idx = []
        self.labels = []
        self.is_target_label = None
        self.scaler = StandardScaler()

    def train_valid_split(self, valid_list):
        if sum(valid_list) != 1:
            raise ValueError('train_valid_split doesnt sum up to 1')
        if max(valid_list) > valid_list[0]:
            raise ValueError('Train data set is smaller than validation data sets (Note: First percentage gives size of train data set)')
        self.train_split_idx = [0]
        total_size = 0
        for size in valid_list:
            total_size = total_size + size
            self.train_split_idx.append(int(len(self.x_train)*total_size))
            if self.train_split_idx[1] <= 0:
                raise ValueError('train data set too small')
        self.split_valid_train_set()

    def split_valid_train_set(self):
        self.x_valid = [[]] * (len(self.train_split_idx) - 2)
        self.y_valid = [[]] * (len(self.train_split_idx) - 2)
        for idx_counter, __ in enumerate(self.train_split_idx[:-1]):
            if idx_counter > 0:
                first_index = self.train_split_idx[idx_counter]
                second_index = self.train_split_idx[idx_counter + 1]
                self.x_valid[idx_counter - 1] = self.x_train[first_index:second_index,:]
                self.y_valid[idx_counter - 1] = self.y_train[first_index:second_index,:]
        self.x_train = self.x_train[self.train_split_idx[0]:self.train_split_idx[1], :]
        self.y_train = self.y_train[self.train_split_idx[0]:self.train_split_idx[1], :]

    def __str__(self):
        out_str = 'Data Class containing: \n'
        out_str+='Scaler: '+str(self.scaler)+'\n'
        out_str+='Labels: '+str(self.labels)+'\n'
        return(out_str)

    def __repr__(self):
        return self.__str__()
