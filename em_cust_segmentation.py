# Library importing

import numpy as np
import pandas as pd
import sklearn
from sklearn import set_config
set_config(transform_output = "pandas")
from sklearn.cluster import KMeans
import statsmodels.api as sm

# Data importing

em_segmentation = pd.read_pickle('C:/Users/Usuario/Desktop/Proyects/Easy Money/EasyMoney_/pickles/em_segmentation.pkl')
em_segmentation_trans = pd.read_pickle('C:/Users/Usuario/Desktop/Proyects/Easy Money/EasyMoney_/pickles/em_segmentation_trans.pkl')

# Functions

def calc_moda(series):
    return series.mode().iloc[0]

def clustering(df_train, df_, n_clusters):
  cluster = KMeans(n_clusters, random_state = 42)
  cluster.fit(df_train)
  labels = cluster.predict(df_train)
  df_['cluster'] = labels
  return df_

def generate_multiindex(list_of_tuples, names):
    return pd.MultiIndex.from_tuples(list_of_tuples, names = names)

# Modelling 

em_segmentation_trans.drop('ordinal__region_code', axis=1, inplace= True)
em_clustered = clustering(em_segmentation_trans, em_segmentation, n_clusters = 4)

# Results analysis

em_clustered.groupby('cluster').agg(
    average_age = ('age','mean'),
    average_salary = ('salary','mean'),
    average_net_margin = ('T_net_margin','mean'),
    average_sales = ('T_sales','mean'),
    average_afiliation_days = ('afiliation_days','mean'),
    entry_channel = ('entry_channel_most_freq', calc_moda),
    segment = ('segment', calc_moda),
    customer_seniority = ('customer_seniority', calc_moda)).T

numeric_desc = []
for col in ["age", "salary", "T_net_margin", "T_sales", 'afiliation_days']:
    data_summary = em_clustered[["cluster", col]].groupby("cluster").describe().T[1:]
    numeric_desc.append(data_summary)
num_file = pd.concat(numeric_desc)

cat_desc = []
for col in ["entry_channel_most_freq", "segment", "customer_seniority"]:
    data_summary = em_clustered[["cluster", col]].groupby("cluster").describe().T[1:]
    cat_desc.append(data_summary)
cat_file = pd.concat(cat_desc)

# Building multiindex

inner_index = [
    "Age",
    "Salary",
    "Net margin",
    "Number of sales",
    'Affiliation days'
]

Statisticals = ["Mean", "St desviation", "Min", "Perc. 25", "Perc. 50", "Perc. 75", "Max"]
new_multi_index = []

for ii, in zip(inner_index):
    for es in Statisticals:
        new_multi_index.append((ii, es))

names = ["Indicator", "Statistical"]
index_file = generate_multiindex(new_multi_index, names)
num_file.set_index(index_file, inplace = True)

clusters_size = em_clustered.groupby("cluster").size().to_frame().T
clusters_size.set_index(generate_multiindex([("Cluster", "Size")] , names), inplace = True)
num_file.groupby(level=0).first().T

num_file = pd.concat([num_file, clusters_size])

num_file = num_file.rename(columns = {
    0: "Linked, younger age, lower salary, lower sales, and lower profit",
    1: "Unlinked, younger age, good salary, good sales, and medium profit",
    2: "Linked, older age, good salary, medium sales, and good profit",
    3: "Unlinked, older age, lower salary, higher sales, and higher profit",
})

# Pickles
pd.to_pickle(em_clustered, 'C:/Users/Usuario/Desktop/Proyects/Easy Money/EasyMoney_/pickles/em_clustered_benchmark')
pd.to_pickle(num_file, 'C:/Users/Usuario/Desktop/Proyects/Easy Money/EasyMoney_/pickles/cluster_file')