"""
测试
"""
import os

import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from ewkmeans import EWKmeans

os.chdir("/home/lyh/MyDocument/Direction/code/ewkm/scikit_learn_data/")
twenty_train = load_files('20news-home/train')
twenty_data = twenty_train.data

count_vect = CountVectorizer(stop_words='english', decode_error='ignore')
tfidf_transformer = TfidfTransformer()
train_counts = count_vect.fit_transform(twenty_data)
train_tfidf = tfidf_transformer.fit_transform(train_counts)
each_dimen = np.sum(train_tfidf, axis=0)
each_dimen_sort = np.argsort(each_dimen)
print(each_dimen_sort)
print(each_dimen[0][0])
EWK = EWKmeans(n_cluster=20, gamma=10.0)

y1_pre = EWK.fit_predict(train_tfidf)
