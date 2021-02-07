import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


class Scaler:
    def __init__(self):
        self.pipeline = Pipeline([('standard', StandardScaler()), ('min_max', MinMaxScaler())])
        self.flag = False
    
    def scale(self, data):
        assert isinstance(data, pd.DataFrame)
        self.columns = list(data.columns)
        data = self.pipeline.fit_transform(data.values)
        self.flag = True
        
        return data, self

    def inverse_scale(self, data, scaler=None):
        assert isinstance(data, np.ndarray)
        if self.flag:
            data = self.pipeline.inverse_transform(data)
        else:
            assert scaler is not None
            data = scaler.inverse_transform(data)

        data = pd.DataFrame(data, columns=self.columns)

        return data


def matching_indices(a, b):
    """Returns indices of elements in b which are elements in a"""
    return [i for i, item in enumerate(a) if item in b]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


class MultiColumnLabelEncoder:
    def __init__(self,columns):
        self.columns = columns

    def fit(self, X):
        self.encoders = defaultdict(None)
        for col in self.columns:
            if col in X.columns:
                le = LabelEncoder()
                le.fit(X[col])
                self.encoders[col] = le

        return self

    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        assert hasattr(self, 'encoders'), "Please first fit() before transform()"

        output = X.copy()
        for col in self.columns:
            if col in output.columns:
                le = LabelEncoder()
                output[col] = le.fit_transform(output[col])
                self.encoders[col] = le

        return output
    
    def inverse_transform(self, X):
        assert hasattr(self, 'encoders'), "Please first run encoding before running inverse encoding"
        
        output = X.copy()

        for col in self.columns:

            if col in output.columns:
                le = self.encoders[col]
                available_values = list(np.unique(output[col]))
                possible_values = list(le.transform(le.classes_))

                if not set(available_values).issubset(set(possible_values)):
                    out_values = [x for x in available_values if x not in possible_values]
                    indices_to_be_replaced = []
                    values_to_be_used_to_replace = []
                    for value in out_values:
                        list_ = list(np.where(np.array(output[col]) == value)[0])
                        nearest_feasible = find_nearest(np.array(possible_values), value)
                        indices_to_be_replaced += list_
                        values_to_be_used_to_replace += [nearest_feasible for _ in range(len(list_))]
                    output[col].iloc[indices_to_be_replaced] = values_to_be_used_to_replace

                output[col] = output[col].astype(int)
                output[col] = le.inverse_transform(output[col])
        
        return output


    def fit_transform(self,X):
        return self.fit(X).transform(X)
