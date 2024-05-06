# Library imnmporting

import pandas as pd 
import numpy as np
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn import set_config
set_config(transform_output = "pandas")

# Data importing

PATH = r'C:\Users\Usuario\Desktop\Proyects\Easy Money\data_compressed'
file_name =  r"\customer_commercial_activity.csv"
file_name1 = r"\customer_products.csv"
file_name2 = r"\customer_sociodemographics.csv"
file_name3 = r"\product_description.csv"
file_name4 = r"\sales.csv"
cca = pd.read_csv(PATH + file_name, sep = ",", index_col=0)
cp = pd.read_csv(PATH + file_name1, sep = ",", index_col=0)
cs = pd.read_csv(PATH + file_name2, sep = ",", index_col=0)
prd = pd.read_csv(PATH + file_name3, sep = ",", index_col=0)
sales = pd.read_csv(PATH + file_name4, sep = ",", index_col=0)

# Functions

def calc_moda(series):
    return series.mode().iloc[0]

def setOthers(dataframe, column, num_values, fillvalue):
    top_categories = dataframe[column].value_counts().head(num_values)
    top_categories_list = top_categories.index.to_list()
    top_categories_list.append(fillvalue)
    dataframe[column] = pd.Categorical(dataframe[column], categories=top_categories_list)
    return dataframe[column].fillna(fillvalue)

def seniority(df, date_column, threshold_date):
    df[date_column] = pd.to_datetime(df[date_column])

    df['customer_seniority'] = 'Old'
    df.loc[df.groupby('pk_cid')[date_column].transform('min') >= threshold_date, 'customer_seniority'] = 'New'

    return df

# ------------------------------------
def get_sales_data(sales, prd):
    prd = prd.rename(columns={'pk_product_ID': 'product_ID'})
    sales = sales.rename(columns={'cid' : 'pk_cid'})
    sales['month_sale_int'] = pd.to_datetime(sales['month_sale']).dt.strftime('%Y%m%d').astype(int)
    sales_prd = sales.merge(prd, on= 'product_ID', how= 'inner')

    sales_prd_merge = sales_prd.groupby('pk_cid').agg(
    T_sales = ('pk_sale', 'count'),
    n_product = ('product_desc', 'count'),
    T_net_margin = ('net_margin', 'sum'),
    Mean_net_margin = ('net_margin', 'mean'),
    first_sale = ('month_sale_int', 'min'),
    last_sale = ('month_sale_int', 'max'),
    product_moda = ('product_ID', calc_moda))
    
    return sales_prd_merge

def cust_top_features(df):
    customers_merge = df.groupby('pk_cid').agg(
        entry_channel_nunique = ('entry_channel_so', pd.Series.nunique),
        entry_channel_most_freq = ('entry_channel_so', calc_moda),
        act_cust_most_freq = ('active_customer', 'mean'),
        act_cust_std = ('active_customer', 'std'),
        afiliation_time = ('afiliation_time', 'max'),
        salary = ('salary_imp', 'mean'),
        age = ('age', 'max'),
        region_code = ('region_code_so', calc_moda),
        entry_date = ('entry_date', 'min'),
        segment = ('segment_so', calc_moda),
        customer_seniority = ('customer_seniority', calc_moda)
        )
    customers_merge['act_cust_std'] = customers_merge['act_cust_std'].fillna(-1)
    return customers_merge

def get_customers_data(cca, cs):
        cust = cca.merge(cs, on = ['pk_cid', 'pk_partition'], how = 'inner')

        cust = cust[cust['deceased']== 'N'].drop('deceased', axis =  1)
        cust_es = pd.DataFrame(cust[cust['country_id'] == 'ES'].drop('country_id', axis = 1))

        cust_es['active_customer'] = cust_es['active_customer'].astype(int)
        cust_es['pk_partition'] = pd.to_datetime(cust_es['pk_partition'])
        cust_es['entry_date'] = pd.to_datetime(cust_es['entry_date'])
        cust_es['afiliation_time'] = cust_es['pk_partition'] - cust_es['entry_date']

        cust_es['gender_mf'] = SimpleImputer(strategy =  'most_frequent').fit_transform(
                cust_es[['gender']])
        sal_mn_regage = cust_es.groupby(['age', 'region_code'])['salary'].transform('mean')
        cust_es['salary_imp'] = cust_es['salary'].fillna(sal_mn_regage).round(2)
        cust_es['salary_imp'] = SimpleImputer(strategy = 'mean').fit_transform(cust_es[['salary_imp']])

        cust_es['segment_so'] = setOthers(cust_es, 'segment', 2, '03 - Other')
        cust_es['entry_channel_so'] = setOthers(cust_es, 'entry_channel', 10, 'ZZZ')
        cust_es['region_code_so'] = setOthers(cust_es, 'region_code', 45, 99.0)
        threshold_date = pd.to_datetime('2018-01-01')
        cust_es = seniority(cust_es, 'entry_date', threshold_date)
        return cust_top_features(cust_es)

def getproductdata(dataset):
  dataset.fillna(0, inplace = True)
  #dataset['pk_partition'] = pd.to_datetime(dataset['pk_partition']).dt.strftime('%Y%m%d').astype(int)
  products = dataset.drop(['pk_cid', 'pk_partition'], axis=1).columns
  dataset['Products_holding_sum'] = dataset[products].sum(axis = 1)
  dataset['Products_contracted'] = dataset.select_dtypes(include = np.number).sum(axis = 1)
  cust_products_merge = dataset.groupby('pk_cid').agg(
    mean_products_contracted = ('Products_contracted','mean'),
    std_products_contracted = ('Products_contracted','std'),
    max_products_contracted = ('Products_contracted','max'),
    min_products_contracted = ('Products_contracted','min'))
  cust_products_merge['std_products_contracted'].fillna(-1, inplace = True)

  return cust_products_merge

def mergedataset(cust_es_merge, cust_products_merge, sales_prd_merge):
  full_df = pd.merge(cust_es_merge, cust_products_merge, how = 'inner', left_index = True, right_index = True)
  full_df = pd.merge(full_df, sales_prd_merge, how = 'left', left_index = True, right_index = True)

  full_df['afiliation_days'] = full_df['afiliation_time'].dt.days
  full_df.drop(columns = 'afiliation_time', axis = 1, inplace = True)
  full_df['act_cust_most_freq'] = full_df['act_cust_most_freq'].astype(float)
  full_df.fillna(0, inplace = True)
  full_df['entry_date_int'] = pd.to_datetime(full_df['entry_date']).dt.strftime('%Y%m%d').astype(int)
  full_df.drop('entry_date', axis= 1, inplace = True)

  return full_df

def prep_transform(df):
  #numerical_columns_v2 = df.select_dtypes(exclude = object).drop(['region_code', 'product_moda'], axis=1).columns.to_list()
  numerical_columns_v2 = df.select_dtypes(exclude = object).drop('region_code', axis=1).columns.to_list()
  transform_pipe = ColumnTransformer(transformers = [
    ("scaler", MinMaxScaler(), numerical_columns_v2),
    ("encoder", OneHotEncoder(sparse_output = False), ['entry_channel_most_freq', 'segment', 'product_moda']),
    #("encoder", OneHotEncoder(sparse_output = False), ['entry_channel_most_freq', 'segment', 'product_moda']),
    ('ordinal', OrdinalEncoder(), ['region_code', 'customer_seniority'])
    ])
  full_df_trans = transform_pipe.fit_transform(df)
  return full_df_trans

# Preprocessing

sales_prd_merge = get_sales_data(sales, prd)
cust_es_merge = get_customers_data(cca, cs)
cust_products_merge =getproductdata(cp)

full_df = mergedataset(cust_es_merge, cust_products_merge, sales_prd_merge)
full_df_trans = prep_transform(full_df)

em_propensity = full_df_trans.drop_duplicates()
prd_prop_to_merge = cp.drop('pk_partition', axis=1).drop_duplicates().set_index('pk_cid')
em_propensity_pp = pd.merge(em_propensity, prd_prop_to_merge[['Products_contracted','Products_holding_sum', 'pension_plan']], 
                            how = 'left', left_index = True, right_index = True).drop_duplicates()
em_propensity_emc = pd.merge(em_propensity, prd_prop_to_merge[['Products_contracted','Products_holding_sum', 'emc_account']], 
                            how = 'left', left_index = True, right_index = True).drop_duplicates()
em_propensity_dc = pd.merge(em_propensity, prd_prop_to_merge[['Products_contracted','Products_holding_sum', 'debit_card']], 
                            how = 'left', left_index = True, right_index = True).drop_duplicates()

# Pickles

pd.to_pickle(em_propensity_pp,"C:/Users/Usuario/Desktop/Proyects/Easy Money/EasyMoney_/pickles/em_propensity_pp")
pd.to_pickle(em_propensity_emc,"C:/Users/Usuario/Desktop/Proyects/Easy Money/EasyMoney_/pickles/em_propensity_emc")
pd.to_pickle(em_propensity_dc,"C:/Users/Usuario/Desktop/Proyects/Easy Money/EasyMoney_/pickles/em_propensity_dc")

