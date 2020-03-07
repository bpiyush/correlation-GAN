import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier

from data.preprocess import MultiColumnLabelEncoder
from utils.data import check_data_type
from utils.metrics import corr, cramers_v, batchwise_editdistance


class Reidentification(object):
    """docstring for Reidentification"""
    def __init__(self, metric='euclidean', epsilon=0.3):
        super(Reidentification, self).__init__()
        self.metric = metric
        self.epsilon = epsilon

    def estimate_unknowns(self, target_known_info, released_dataset, qid_columns, non_qid_columns):
        nbrs = NearestNeighbors(n_neighbors=1)
        nbrs.fit(released_dataset[qid_columns].values)

        dist, idx = nbrs.kneighbors(target_known_info.values)
        idx_list = idx[:, 0]

        estimated_unknown_info = released_dataset.iloc[idx_list][non_qid_columns]

        return estimated_unknown_info
    
    
    def compute_distance(self, actual_target_records, estimated_target_records, num_real):

        actual_real_part = actual_target_records.iloc[:, :num_real].astype(float)
        estimated_real_part = estimated_target_records.iloc[:, :num_real].astype(float)
        
        scaler = StandardScaler()
        scaler.fit(pd.concat([actual_real_part, estimated_real_part]).values)
        
        actual_real_part = scaler.transform(actual_real_part)
        estimated_real_part = scaler.transform(estimated_real_part)
        
        real_distances = np.linalg.norm(actual_real_part - estimated_real_part, axis=1)
        
        if num_real < actual_target_records.shape[1]:
            actual_cat_part = actual_target_records.iloc[:, num_real:]
            estimated_cat_part = estimated_target_records.iloc[:, num_real:]

            cat_distances = batchwise_editdistance(actual_cat_part, estimated_cat_part)

            real_distances = real_distances / (real_distances.max() - real_distances.min() + 1e-4)
            cat_distances = cat_distances / (cat_distances.max() - cat_distances.min() + 1e-4)

            return (real_distances + cat_distances) / 2.0
        else:
            return real_distances / (real_distances.max() - real_distances.min() + 1e-4)

    
    def reidentify(self, released_dataset, target_records, qid_columns, num_real):

        cat_columns = released_dataset.columns[num_real:]
        mLE = MultiColumnLabelEncoder(columns=list(cat_columns))
        
        combined_dataset = pd.concat([released_dataset, target_records])
        mLE.fit(combined_dataset)

        released_dataset_encoded = mLE.transform(released_dataset)

        all_columns = target_records.columns
        non_qid_columns = list(set(all_columns) - set(qid_columns))

        target_known_info = mLE.transform(target_records[qid_columns])
        actual_target_unknown_info = mLE.transform(target_records[non_qid_columns])
        estimated_target_unknown_info = self.estimate_unknowns(target_known_info, released_dataset_encoded,
                                                               qid_columns, non_qid_columns)
        
        actual_target_records = mLE.transform(target_records)
        estimated_target_records = actual_target_records.copy()
        estimated_target_records[qid_columns] = target_known_info
        estimated_target_records[non_qid_columns] = estimated_target_unknown_info.values

        actual_target_records = mLE.inverse_transform(actual_target_records)
        estimated_target_records = mLE.inverse_transform(estimated_target_records)

        distances = self.compute_distance(actual_target_records, estimated_target_records, num_real)
        
        return distances


class AttributeDisclosure(object):
    """docstring for AttributeDisclosure"""
    def __init__(self):
        super(AttributeDisclosure, self).__init__()
    
    def disclose_sensitive_info(self, released_dataset, target_records,
                                sensitive_columns, qid_columns, num_real, num_ngbrs):
        all_columns = target_records.columns
        non_qid_columns = list(set(all_columns) - set(qid_columns))
        
        real_sensitive_columns = [x for x in sensitive_columns if check_data_type(x, all_columns, num_real) == 'real']
        cat_sensitive_columns = list(set(sensitive_columns) - set(real_sensitive_columns))

        cat_columns = released_dataset.columns[num_real:]
        mLE = MultiColumnLabelEncoder(columns=list(cat_columns))
        combined_dataset = pd.concat([released_dataset, target_records])
        mLE.fit(combined_dataset)
        
        released_dataset_encoded = mLE.transform(released_dataset)
        target_records_encoded = mLE.transform(target_records)

        actual_target_records = target_records_encoded.copy()
        target_records_known = target_records_encoded[qid_columns]

        num_records, num_features = actual_target_records.shape
        num_qids = len(qid_columns)

        corr_arr = defaultdict(list)
        final_estimation_success = 0.0

        if len(real_sensitive_columns) > 0:
            knnReg = KNeighborsRegressor(n_neighbors=num_ngbrs)

            knnReg.fit(released_dataset_encoded[qid_columns], released_dataset_encoded[real_sensitive_columns])
            estimated_sensitive_info_real = knnReg.predict(target_records_known)
            actual_sensitive_info_real = actual_target_records[real_sensitive_columns].values
            
            estimation_success = corr(estimated_sensitive_info_real, actual_sensitive_info_real)
            corr_arr['real'].extend(estimation_success)
            
            fraction_real = (len(real_sensitive_columns) / len(sensitive_columns))
            final_estimation_success += np.array(corr_arr['real']).mean() * fraction_real

        if len(cat_sensitive_columns) > 0:
            knnClf = KNeighborsClassifier(n_neighbors=num_ngbrs)

            knnClf.fit(released_dataset_encoded[qid_columns], released_dataset_encoded[cat_sensitive_columns])
            estimated_sensitive_info_cat = knnReg.predict(target_records_known)
            actual_sensitive_info_cat = actual_target_records[cat_sensitive_columns].values
            corr_arr['cat'] = np.zeros(len(cat_sensitive_columns))

            for i in range(len(cat_sensitive_columns)):
                corr_value = cramers_v(estimated_sensitive_info_cat.iloc[:, i], actual_sensitive_info_cat.iloc[:, i])
                corr_arr['cat'][i] = corr_value if not np.isnan(corr_value) else 0.0
            
            fraction_cat = (len(cat_sensitive_columns) / len(sensitive_columns))
            final_estimation_success += np.array(corr_arr['cat']).mean()  * fraction_cat

        return final_estimation_success, corr_arr


