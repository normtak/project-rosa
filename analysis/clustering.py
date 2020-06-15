# -*- coding: utf-8 -*-
"""
Created on Tue May 19 22:17:17 2020

@author: Stan
"""

import pandas as pd
import numpy as np
import sys
sys.path.append(r'C:\workdir\rosabella\project-rosa\modules')
import modules
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

#Load and process data
df = pd.read_excel(r'C:\workdir\rosabella\sales.xlsx')
df.columns = ['date_time', 'item_name', 'id', 'sales_qtt']
sep = ','
df['item_name_cut'] = df['item_name'].str.split(',').str[0]
df['item_name_cut'] = df['item_name_cut'].str.replace('\u200bОчищающая плитка "Dolce"', 'Очищающая плитка "Dolce"')
df['item_name_cut'] = df['item_name_cut'].str.replace('(миниатюра)', '', regex=False)
#df['item_name_cut'] = df['item_name_cut'].str.split().str[:2]
#df['item_name_cut'] = df['item_name_cut'].apply(lambda x: " ".join(x))

#unigram
allnames = list(df.item_name_cut.drop_duplicates())
bag_vectors_list = modules.generate_bow(allnames)
bag_vectors_arr = np.array(bag_vectors_list)

kmeans = KMeans(n_clusters=15)
kmeans.fit(bag_vectors_arr)
kmeans.cluster_centers_
kmeans.labels_
a = list(kmeans.labels_)

#bi-gram
vectorizer = CountVectorizer(ngram_range=(1,1))
X = vectorizer.fit_transform(allnames)
print(vectorizer.get_feature_names())
print(X.toarray().shape)
bag_vectors_arr_bigram = X.toarray()

kmeans_bi = KMeans(n_clusters=10)
kmeans_bi.fit(bag_vectors_arr_bigram)
kmeans_bi.cluster_centers_
kmeans_bi.labels_
a = list(kmeans_bi.labels_)


s = "My girlfriend is a rich lady"
print(" ".join(s.split()[:2]))
p = s.split()[:2]
