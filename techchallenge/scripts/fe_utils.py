#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from datetime import timedelta
import gc

# In[ ]:


#-------------Functions

#------- Standard columns
def standard_column_names(df_pd):
  """
  Funtion to standardize column names Upper case the names, Replace spaces with underscore
  (,) with blanks,% with PER and - with undescore
  df: Input dataframe
  df_pd: Output with Standardzied column names
  """
  #df_pd=df.toPandas()
  df_pd.columns = df_pd.columns.str.strip().str.upper().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('%', 'PER').str.replace('-', '_')
  return df_pd

#------ date features
def build_temporal_features(data: pd.DataFrame,col) -> pd.DataFrame:
    # Temporal features
    data[col] = pd.to_datetime(data[col])
    data['year'] = data[col].dt.year
    data['month'] = data[col].dt.month.astype('str').str.zfill(2)
    data['quarter'] = data[col].dt.quarter
    data['week'] = data[col].dt.week.astype('str').str.zfill(2)
    #data['day'] = data[col].dt.day
    #data['dayofweek'] = data[col].dt.dayofweek
    #data['week_of_month'] = data['day'].apply(lambda x: np.ceil(x / 7)).astype(np.int8)
    #data['is_weekend'] = (data['dayofweek'] > 5).astype(np.int8)
    #data['quarter'] = data[col].dt.quarter
    # Calculate the quarter:
    #data['is_quarter_start'] = data[col].dt.is_quarter_start
    # Mapping the value (True = 1 and False = 0):
    #data['is_quarter_start'] = data['is_quarter_start'].map({True: 1, False:0})
    # Calculate the quarter end:
    #data['is_quarter_end'] = data[col].dt.is_quarter_end
    # Mapping the value (True = 1 and False = 0):
    #data['is_quarter_end'] = data['is_quarter_end'].map({True: 1, False:0})
    # Calculate the day:
    #data['is_month_start'] = data[col].dt.is_month_start
    # Mapping the value (True = 1 and False = 0):
    #data['is_month_start'] = data['is_month_start'].map({True: 1, False:0})
    # Calculate the day:
    #data['is_month_end'] = data[col].dt.is_month_end
    # Mapping the value (True = 1 and False = 0):
    #data['is_month_end'] = data['is_month_start'].map({True: 1, False:0})
    # Calculate the day:
    #data['is_leap_year'] = data[col].dt.is_leap_year
    # Mapping the value (True = 1 and False = 0):
    #data['is_leap_year'] = data['is_leap_year'].map({True: 1, False:0})
    
    
    
    # First and Latest SALES_VOLUME Date
    #data_temp = data.copy()
    #data_temp = data_temp[data_temp[col]>0]
    #data['First SALES_VOLUME'] = data_temp['Posting Date'].min()
    #data['Latest SALES_VOLUME'] = data_temp['Posting Date'].max()    
    
    #data['week_start'] = data[col] - data[col].dt.weekday.astype('timedelta64[D]')
    #data['week_end'] = data['week_start'] + timedelta(days=6)
    return data

#----- Lag Features
def lag_feature(df, lags, col,group_by_columns_order_lags,lag_type):
    list_col=list([col])
    new_list=list(set(group_by_columns_order_lags + list_col))
    #print(new_list)
    df_temp=df
    for i in lags:
        shifted = df[new_list]
        #list_col_tmp=list([col+'_LAG_'+str(i)])
        #new_list_tmp=list(set(group_by_columns_order_lags + list_col_tmp))
        shifted.rename(columns={col:col+'_LAG_'+str(i)}, inplace=True)
        #shifted.columns = new_list_tmp
        #print(shifted.dtypes)
        shifted[lag_type]=shifted[lag_type].apply(lambda x:x+int(i))
        df_temp = pd.merge(df_temp, shifted, on=group_by_columns_order_lags, how='left')
    return df_temp



#-------------Rolling Feature
def create_qty_window_features(df,group_by_columns_orders, order_lag_column,lag_params):
    if lag_params['activate']:
        lag = lag_params['lag']
        index_columns=lag_params['index_columns']
        #print(f'shift_lag: {lag}')
        if lag_params['lag_feature'] == True:
            df.loc[:,f'lag_l{lag}'] = df.sort_values(group_by_columns_orders+list([index_columns])).groupby(group_by_columns_orders)[order_lag_column].transform(lambda x: x.shift(lag))
        if lag_params['roll_mean_win'] != []:
            for win in lag_params['roll_mean_win']:
                #print(f'rolling_mean: {lag}/{win}')                                                                       
                df[f'rolling_mean_l{lag}_w{win}'] = df.sort_values(group_by_columns_orders+list([index_columns])).groupby(group_by_columns_orders)[order_lag_column].transform(lambda x: x.shift(lag).rolling(win).mean())
        if lag_params['roll_median_win'] != []:
            for win in lag_params['roll_median_win']:
                #print(f'rolling_median: {lag}/{win}')                                                                       
                df[f'rolling_median_l{lag}_w{win}'] = df.sort_values(group_by_columns_orders+list([index_columns])).groupby(group_by_columns_orders)[order_lag_column].transform(lambda x: x.shift(lag).rolling(win).median())
        if lag_params['roll_std_win'] != []:
            for win in lag_params['roll_std_win']:
                #print(f'rolling_std: {lag}/{win}')                                                                       
                df[f'rolling_std_l{lag}_w{win}'] = df.sort_values(group_by_columns_orders+list([index_columns])).groupby(group_by_columns_orders)[order_lag_column].transform(lambda x: x.shift(lag).rolling(win).std())
    df=standard_column_names(df)
    return df

params_lag4_win2 = {'activate': False,
               'lag': 4,
               'lag_feature':True,
               'roll_mean_win':[2],
               'roll_median_win':[],
               'roll_std_win':[2],
               'index_columns':'ROLLING_WEEK'}

params_lag8_win2 = {'activate': False,
               'lag': 8,
               'lag_feature':True,
               'roll_mean_win':[2],
               'roll_median_win':[],
               'roll_std_win':[2],
               'index_columns':'ROLLING_WEEK'}

params_lag12_win2 = {'activate': False,
               'lag': 12,
               'lag_feature':True,
               'roll_mean_win':[2],
               'roll_median_win':[],
               'roll_std_win':[2],
               'index_columns':'ROLLING_WEEK'}

params_lag16_win2 = {'activate': False,
               'lag': 16,
               'lag_feature':True,
               'roll_mean_win':[2],
               'roll_median_win':[],
               'roll_std_win':[2],
               'index_columns':'ROLLING_WEEK'}

params_lag20_win2 = {'activate': False,
               'lag': 20,
               'lag_feature':True,
               'roll_mean_win':[2],
               'roll_median_win':[],
               'roll_std_win':[2],
               'index_columns':'ROLLING_WEEK'}
params_lag24_win2 = {'activate': False,
               'lag': 24,
               'lag_feature':True,
               'roll_mean_win':[2],
               'roll_median_win':[],
               'roll_std_win':[2],
               'index_columns':'ROLLING_WEEK'}
# ------- window=4
params_lag4_win4 = {'activate': False,
               'lag': 4,
               'lag_feature':True,
               'roll_mean_win':[4],
               'roll_median_win':[],
               'roll_std_win':[4],
               'index_columns':'ROLLING_WEEK'}

params_lag8_win4 = {'activate': False,
               'lag': 8,
               'lag_feature':True,
               'roll_mean_win':[4],
               'roll_median_win':[],
               'roll_std_win':[4],
               'index_columns':'ROLLING_WEEK'}

params_lag12_win4 = {'activate': False,
               'lag': 12,
               'lag_feature':True,
               'roll_mean_win':[4],
               'roll_median_win':[],
               'roll_std_win':[4],
               'index_columns':'ROLLING_WEEK'}

params_lag16_win4 = {'activate': False,
               'lag': 16,
               'lag_feature':True,
               'roll_mean_win':[4],
               'roll_median_win':[],
               'roll_std_win':[4],
               'index_columns':'ROLLING_WEEK'}

params_lag20_win4 = {'activate': False,
               'lag': 20,
               'lag_feature':True,
               'roll_mean_win':[4],
               'roll_median_win':[],
               'roll_std_win':[4],
               'index_columns':'ROLLING_WEEK'}
params_lag24_win4 = {'activate': False,
               'lag': 24,
               'lag_feature':True,
               'roll_mean_win':[4],
               'roll_median_win':[],
               'roll_std_win':[4],
               'index_columns':'ROLLING_WEEK'}


#-------------Mean Encoding

##'CHANNEL','COUNTRY','PRODUCT_SEASON', 'PRODUCT_GROUP',
##       'PRODUCT_SUBGROUP', 'PRODUCT_CLASS', 'PRODUCT_SUBCLASS',
#params_mean_encoding_product_subgroup_week
#params_mean_encoding_product_group_week
#params_mean_encoding_product_class_week
#params_mean_encoding_product_subclass_week
params_mean_encoding_product_subgroup_week = {'activate': True,
                         'group_by_columns': ["PRODUCT_SUBGROUP","ROLLING_WEEK"],
                         'agg_column':'SALES_VOLUME_LAG_1',
                         'column_name':'SALES_VOLUME_PRODUCT_SUBGROUP',
                         'no_of_lags':[1,2,3,4],
                         'group_by_columns_order_lags':['PRODUCT_SUBGROUP','ROLLING_WEEK'],
                         'lag_type':['ROLLING_WEEK'],
                         'merge_column_name':'_mean_encoding_PRODUCT_SUBGROUP'}
params_mean_encoding_product_group_week = {'activate': True,
                         'group_by_columns': ["PRODUCT_GROUP","ROLLING_WEEK"],
                         'agg_column':'SALES_VOLUME_LAG_1',
                         'column_name':'SALES_VOLUME_PRODUCT_GROUP',
                         'no_of_lags':[1,2,3,4],
                         'group_by_columns_order_lags':['PRODUCT_GROUP','ROLLING_WEEK'],
                         'lag_type':['ROLLING_WEEK'],
                         'merge_column_name':'_mean_encoding_PRODUCT_GROUP'}
params_mean_encoding_product_class_week = {'activate': True,
                         'group_by_columns': ["PRODUCT_CLASS","ROLLING_WEEK"],
                         'agg_column':'SALES_VOLUME_LAG_1',
                         'column_name':'SALES_VOLUME_PRODUCT_CLASS',
                         'no_of_lags':[1,2,3,4],
                         'group_by_columns_order_lags':['PRODUCT_CLASS','ROLLING_WEEK'],
                         'lag_type':['ROLLING_WEEK'],
                         'merge_column_name':'_mean_encoding_PRODUCT_CLASS'}
params_mean_encoding_product_subclass_week = {'activate': True,
                         'group_by_columns': ["PRODUCT_SUBCLASS","ROLLING_WEEK"],
                         'agg_column':'SALES_VOLUME_LAG_1',
                         'column_name':'SALES_VOLUME_PRODUCT_SUBCLASS',
                         'no_of_lags':[1,2,3,4],
                         'group_by_columns_order_lags':['PRODUCT_SUBCLASS','ROLLING_WEEK'],
                         'lag_type':['ROLLING_WEEK'],
                         'merge_column_name':'_mean_encoding_PRODUCT_SUBCLASS'}


def create_mean_encodings(df,mean_encoding_params):
    if mean_encoding_params['activate']:
        group_by_columns=mean_encoding_params['group_by_columns']
        agg_column=mean_encoding_params['agg_column']
        column_name=mean_encoding_params['column_name']
        no_of_lags=mean_encoding_params['no_of_lags']
        group_by_columns_order_lags=mean_encoding_params['group_by_columns_order_lags']
        lag_type=mean_encoding_params['lag_type']
        merge_column_name = mean_encoding_params['merge_column_name']
    
        ## Group first
        group = df.groupby(group_by_columns).agg({agg_column: ['mean']})
        
        group.columns = list([column_name])
        group.reset_index(inplace=True)
    
        ## Create Lags 
        group = lag_feature(group, no_of_lags, column_name,group_by_columns_order_lags,lag_type)
        #group.rename(columns = {agg_column:'SALES_VOLUME_mean'},inplace=True)
        ## Merge with original data set
        df = pd.merge(df, group, on=group_by_columns, how='left',validate = 'many_to_one',suffixes=['',merge_column_name])
        #df[column_name] = df[column_name].astype(np.float16)
    
        #df.drop([column_name], axis=1, inplace=True)
  
    return df


#--------------------------- Expanding Weight
def create_expanding_weighted_features(df,expanding_params):
    if expanding_params['activate']:
        expand_type=expanding_params['expand_type']
        group_by_columns=expanding_params['group_by_columns']
        agg_column=expanding_params['agg_column']
        column_name=expanding_params['column_name']
        agg_type=expanding_params['agg_type']
        expand_group_by_columns=expanding_params['expand_group_by_columns']
        index_column=expanding_params['index_column']
        expand_column_name=expanding_params['expand_column_name']
        no_weights = expanding_params['no_weights']
        #print(column_name)
        #merge=df.groupby(group_by_columns)[agg_column].agg(agg_type)
        #merge.rename(columns={})
        ##First Group by 
        column_name=str(agg_type)+'_'+column_name
        #print(column_name)
        group = df.groupby(group_by_columns).agg({agg_column: [agg_type]})
        group.columns = list([column_name])
        group.reset_index(inplace=True)
        group.set_index(index_column, inplace = True)
        
        if expand_type=='min':
        # The minimum of the orders in all the previous weeks
            expand = group.groupby(expand_group_by_columns)[column_name].expanding().min().reset_index()
            expand.set_index(index_column, inplace = True)
            expand.rename(columns = {column_name: expand_column_name}, inplace = True)
            expand=standard_column_names(expand)
        elif expand_type=='max':
            expand = group.groupby(expand_group_by_columns)[column_name].expanding().max().reset_index()
            expand.set_index(index_column, inplace = True)
            expand.rename(columns = {column_name: expand_column_name}, inplace = True)
            expand=standard_column_names(expand)
        elif expand_type=='mean':
            expand = group.groupby(expand_group_by_columns)[column_name].expanding().mean().reset_index()
            expand.set_index(index_column, inplace = True)
            expand.rename(columns = {column_name: expand_column_name}, inplace = True)
            expand=standard_column_names(expand)

        elif expand_type=='weighted_average':
            weights_3 = np.arange(1,no_weights+1)
            expand = group.groupby(expand_group_by_columns)[column_name].rolling(no_weights).apply(lambda prices: np.dot(prices, weights_3)/weights_3.sum(), raw=True).reset_index()
            expand.set_index(index_column, inplace = True)
            expand.rename(columns = {column_name: expand_column_name}, inplace = True)
            expand=standard_column_names(expand)

        return expand.reset_index()

params_expanding_min_weighted = {'activate': True,
                         'expand_type':'min',
                         'group_by_columns': ["KEY","ROLLING_WEEK"],
                         'agg_column':'SALES_VOLUME_LAG_1',
                         'column_name':'SALES_VOLUME_LAG_1',
                         'agg_type':'mean',
                         'no_weights':4,
                         'no_of_lags':[2,3,4],
                         'expand_group_by_columns':['KEY'],
                         'index_column':['ROLLING_WEEK'],
                         'expand_column_name':'EXPANDING_MIN_SALES_VOLUME'}

params_expanding_max_weighted = {'activate': True,
                         'expand_type':'max',
                         'group_by_columns': ["KEY","ROLLING_WEEK"],
                         'agg_column':'SALES_VOLUME_LAG_1',
                         'column_name':'SALES_VOLUME_LAG_1',
                         'agg_type':'mean',
                         'no_weights':4,
                         'no_of_lags':[2,3,4],
                         'expand_group_by_columns':['KEY'],
                         'index_column':['ROLLING_WEEK'],
                         'expand_column_name':'EXPANDING_MAX_SALES_VOLUME'}

params_expanding_mean_weighted = {'activate': True,
                         'expand_type':'mean',
                         'group_by_columns': ["KEY","ROLLING_WEEK"],
                         'agg_column':'SALES_VOLUME_LAG_1',
                         'column_name':'SALES_VOLUME_LAG_1',
                         'agg_type':'mean',
                         'no_weights':4,
                         'no_of_lags':[2,3,4],
                         'expand_group_by_columns':['KEY'],
                         'index_column':['ROLLING_WEEK'],
                         'expand_column_name':'EXPANDING_MEAN_SALES_VOLUME'}

params_expanding_mean_weighted_4week = {'activate': True,
                         'expand_type':'weighted_average',
                         'group_by_columns': ["KEY","ROLLING_WEEK"],
                         'agg_column':'SALES_VOLUME_LAG_1',
                         'column_name':'SALES_VOLUME_LAG_1',
                         'agg_type':'mean',
                         'no_weights':4,
                         'no_of_lags':[2,3,4],
                         'expand_group_by_columns':['KEY'],
                         'index_column':['ROLLING_WEEK'],
                         'expand_column_name':'EXPANDING_MEAN_SALES_VOLUME_4WEEK'}

params_expanding_mean_weighted_12week = {'activate': True,
                         'expand_type':'weighted_average',
                         'group_by_columns': ["KEY","ROLLING_WEEK"],
                         'agg_column':'SALES_VOLUME_LAG_1',
                         'column_name':'SALES_VOLUME_LAG_1',
                         'agg_type':'mean',
                         'no_weights':12,
                         'no_of_lags':[2,3,4],
                         'expand_group_by_columns':['KEY'],
                         'index_column':['ROLLING_WEEK'],
                         'expand_column_name':'EXPANDING_MEAN_SALES_VOLUME_12WEEK'}
                         
params_expanding_mean_weighted_8week = {'activate': True,
                         'expand_type':'weighted_average',
                         'group_by_columns': ["KEY","ROLLING_WEEK"],
                         'agg_column':'SALES_VOLUME_LAG_1',
                         'column_name':'SALES_VOLUME_LAG_1',
                         'agg_type':'mean',
                         'no_weights':8,
                         'no_of_lags':[2,3,4],
                         'expand_group_by_columns':['KEY'],
                         'index_column':['ROLLING_WEEK'],
                         'expand_column_name':'EXPANDING_MEAN_SALES_VOLUME_8WEEK'}




def merge_orders_with_expanding(df_orders,df_expanding,merge_on):
    df=pd.merge(df_orders,df_expanding,on=merge_on,how='left',validate='many_to_one')
    df=standard_column_names(df)
    #df_reduced=reduce_memory_usage(df)
    return df

#-----------------Last SALES_VOLUME

# def last_SALES_VOLUME_func(df,TARGET_COLUMN):
    # df=df.fillna(0)
    # b=[0] * len(df)
    # a = np.where(df[TARGET_COLUMN]>0,0,1)
    # #print(a)
    # #print(len(b))
    # for i in range(len(a)):
        # if a[i] == 1:
            # b[i] = b[i-1] + 1
        # else:
            # b[i] = 0
    # #print(b)
    # df['LAST_CONSUMED_SKU'] = b
    # df.LAST_CONSUMED_SKU = np.where((df.ROLLING_WEEK==df.LAST_CONSUMED_SKU) & (df[TARGET_COLUMN]==0),'-1',df.LAST_CONSUMED_SKU)
    # return df


# #--------------- SKU Age
# def age_func(df,TARGET_COLUMN):
    # df=df.fillna('')
    # b=[0] * len(df)
    # c = 1
    # a = np.where(df[TARGET_COLUMN]>0,0,1)
    # t = np.where(a==0)[0].min()
    
    # for i in range(t+1):
        # b[i] = 0
    # for i in range(t+1,len(df)):
        # b[i] = c
        # c = c+1
    
    # df['AGE'] = b
    # return df



################ Feature eng function------------------------------------------------------------
#-------------Controls

#group_by_columns_orders_week = ['KEY','MATERIAL','SBU','TECHNOLOGY_TYPE','YEAR','WEEK'] # add SBU Mat type
#group_by_columns_orders_month = ['KEY','year','month']
#group_by_columns_orders_qusarter = ['KEY','year','quarter']
#group_by_columns_orders_year = ['KEY','year']
NUMBER_OF_LAGS = list(np.arange(1,12))
TARGET_COLUMN = 'SALES_VOLUME'
lag_type_week = 'ROLLING_WEEK'
group_by_columns_orders_lags = ['KEY','PRODUCT_ID','CHANNEL','COUNTRY','PRODUCT_SEASON', 'PRODUCT_GROUP',
       'PRODUCT_SUBGROUP', 'PRODUCT_CLASS', 'PRODUCT_SUBCLASS','ROLLING_WEEK']
group_by_columns_orders = ['KEY','PRODUCT_ID','CHANNEL','COUNTRY','PRODUCT_SEASON', 'PRODUCT_GROUP',
       'PRODUCT_SUBGROUP', 'PRODUCT_CLASS', 'PRODUCT_SUBCLASS']





#------------------- Start-date End-date-------------------------
def get_start_end_dates(year, week):
    d = dt(year,1,1)
    if(d.weekday()<= 3):
        d = d - timedelta(d.weekday())             
    else:
        d = d + timedelta(7-d.weekday())
    dlt = timedelta(days = (week-1)*7)
    return (d + dlt,  d + dlt + timedelta(days=6))




def fe(df):
    #-------------Lag features
    #print('Lags Started')
    df_lag_features = lag_feature(df,NUMBER_OF_LAGS,TARGET_COLUMN,group_by_columns_orders_lags,lag_type_week)
    #print('Lags Completed')
    #print(df_lag_features.shape)
    #df_lag_features.head()
    #------- Rolling Window Features
    #print('Window Features Started')
    df_window=df_lag_features.copy()
    df_window = create_qty_window_features(df_window,group_by_columns_orders, TARGET_COLUMN , params_lag4_win2)
    df_window = create_qty_window_features(df_window,group_by_columns_orders, TARGET_COLUMN , params_lag8_win2)
    df_window = create_qty_window_features(df_window,group_by_columns_orders, TARGET_COLUMN , params_lag12_win2)
    df_window = create_qty_window_features(df_window,group_by_columns_orders, TARGET_COLUMN , params_lag16_win2)
    df_window = create_qty_window_features(df_window,group_by_columns_orders, TARGET_COLUMN , params_lag20_win2)
    df_window = create_qty_window_features(df_window,group_by_columns_orders, TARGET_COLUMN , params_lag24_win2)
    
    df_window = create_qty_window_features(df_window,group_by_columns_orders, TARGET_COLUMN , params_lag4_win4)
    df_window = create_qty_window_features(df_window,group_by_columns_orders, TARGET_COLUMN , params_lag8_win4)
    df_window = create_qty_window_features(df_window,group_by_columns_orders, TARGET_COLUMN , params_lag12_win4)
    df_window = create_qty_window_features(df_window,group_by_columns_orders, TARGET_COLUMN , params_lag16_win4)
    df_window = create_qty_window_features(df_window,group_by_columns_orders, TARGET_COLUMN , params_lag20_win4)
    df_window = create_qty_window_features(df_window,group_by_columns_orders, TARGET_COLUMN , params_lag24_win4)
    #print('Window Features Completed')
    #--- mean Encode
    df_encodings=df_window.copy()
    
    #params_mean_encoding_product_subgroup_week
    #params_mean_encoding_product_group_week
    #params_mean_encoding_product_class_week
    #params_mean_encoding_product_subclass_week

    #print('MEan Encoding Started')
    df_mean_encodings=create_mean_encodings(df_encodings,params_mean_encoding_product_subgroup_week)
    df_mean_encodings=create_mean_encodings(df_mean_encodings,params_mean_encoding_product_group_week)
    df_mean_encodings=create_mean_encodings(df_mean_encodings,params_mean_encoding_product_class_week)
    df_mean_encodings=create_mean_encodings(df_mean_encodings,params_mean_encoding_product_subclass_week)
    #print('MEan Encoding Ended')
    ##Expanding Mean
    group_by_columns_expanding_week=['KEY','ROLLING_WEEK']
    df_expanding=df_mean_encodings.copy()
    #print('Expanding Features Started')
    df_expanding_output_wt_mean_4week=create_expanding_weighted_features(df_expanding,params_expanding_mean_weighted_4week)
    df_expanding_output_wt_mean_12week=create_expanding_weighted_features(df_expanding,params_expanding_mean_weighted_12week)
    df_expanding_output_wt_mean_8week=create_expanding_weighted_features(df_expanding,params_expanding_mean_weighted_8week)
    
    df_expanding_output_min=create_expanding_weighted_features(df_expanding,params_expanding_min_weighted)
    df_expanding_output_max=create_expanding_weighted_features(df_expanding,params_expanding_max_weighted)
    df_expanding_output_mean=create_expanding_weighted_features(df_expanding,params_expanding_mean_weighted)
    #print('Expanding Features Ended')
    df_expand = pd.concat([df_expanding_output_wt_mean_4week,
                           df_expanding_output_wt_mean_12week,
                           df_expanding_output_wt_mean_8week,
                           df_expanding_output_min,
                           df_expanding_output_max,
                           df_expanding_output_mean], axis = 1)
    df_expand = df_expand.loc[:,~df_expand.columns.duplicated()]
    del df_expanding_output_wt_mean_4week
    del df_expanding_output_wt_mean_12week
    del df_expanding_output_wt_mean_8week
    del df_expanding_output_min
    del df_expanding_output_max
    del df_expanding_output_mean
    gc.collect()
    
    df_orders_expanding=merge_orders_with_expanding(df_mean_encodings,df_expand,group_by_columns_expanding_week)
    last_SALES_VOLUME_df = df_orders_expanding.copy()
    #gg=last_SALES_VOLUME_func(last_SALES_VOLUME_df,TARGET_COLUMN)
 
    return last_SALES_VOLUME_df

