## October 2023
## Functions for calculating fitness in the EA
## Author: Claire Little, University of Manchester

## packages needed for the code
import pandas as pd
import numpy as np
import random
from numpy.random import randint, rand
from random import sample
from datetime import datetime
import csv
from joblib import Parallel, delayed
import os
import itertools
#import wquantiles as wq
from operator import add
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit



###############################################################################################################
## Function to calculate the mean squared error of the differences between the proportions
## for a single variable (var_name)
def get_freqs_mse(orig_props, syn, var_name):                      ## input the df with proportions, & synth df
    syn_prop = pd.DataFrame(syn[var_name].value_counts())            ## get counts
    syn_prop[var_name] = syn_prop[var_name]/len(syn)                   ## get proportion
    syn_prop.rename(columns={f"{var_name}": "proportion"}, inplace=True)  ## rename column to match orig
    syn_prop.index.name = 'code'
    ## get mse
    diff = abs(orig_props-syn_prop)               ## difference
    diff['diff_squared'] = diff['proportion']**2  ## squared
    return diff['diff_squared'].mean()            ## return mean


## Function to get the two way proportions between Area and the other variable, and then calculate
## the mean squared error
def get_twoway_mse(orig_props, syn, var_name1, var_name2):
    ## change column name type to int (to be comparable to the syn crosstab table)
    orig_props.columns = orig_props.columns.astype(int) 
    syn_prop = pd.crosstab(syn[var_name1], syn[var_name2])  ## get frequencies
    syn_prop = syn_prop/len(syn)                            ## get proportions
    syn_prop.index.name = 'code'                            ## change index name to match orig
    diff = (abs(orig_props  - syn_prop)**2)                 ## get squared difference
    diff['mean'] = diff.mean(axis=1)                        ## get mean of each row
    return diff['mean'].mean()


def get_freqs_mse_paral(orig_props,population,var_name,joblib_num_cores=-2,joblib_max_nbytes='10M'):
    scores = Parallel(n_jobs=joblib_num_cores,max_nbytes=joblib_max_nbytes)(delayed(get_freqs_mse)(orig_props,_,var_name) for _ in population)
    return scores

def get_twoway_mse_paral(orig_props,population,var_name1,var_name2,joblib_num_cores=-2,joblib_max_nbytes='10M'):
    scores = Parallel(n_jobs=joblib_num_cores,max_nbytes=joblib_max_nbytes)(delayed(get_twoway_mse)(orig_props,_,var_name1,var_name2) for _ in population)
    return scores



###############################################################################################################
## This function (from previous code) takes the synthetic dataframe and transforms the categorical columns.
## It also drops the original column (if drop = True). catcols is a list of the categorical 
## Added later - first it removes rows with NAN (-99, missing) values
def preprocess_cols(df, catcols, numcols, le, scaler, drop_orig_col=False):
    ## remove NAN or -99 values
    df1 = df.copy(deep=True)                  ## make a deep copy
    df1 = df1.replace(-99,np.NaN)             ## replace -99 with NaN
    df1 = df1.dropna(axis=0)                  ## Drop rows with Nan
    df1.reset_index(drop=True,inplace=True)   ## Reset index

    ## categorical columns - previously we treated binary differently but do them all the same
    ## as we need a way to specify the reference level - the only way I can find at the moment is
    ## to get dummy variables for all levels and then drop the reference one after
    for ccol in catcols:
        #n = len(df1[ccol].value_counts())  
        #if (n > 2):   # if it has more than two categories, one-hot encode, dropping the first category
        X = pd.get_dummies(df1[ccol],drop_first=False,prefix=ccol, prefix_sep='_') ## keep first
        df1[X.columns] = X
        if drop_orig_col==True:
            df1.drop(ccol, axis=1, inplace=True)  # drop the original categorical variable (optional)
        #else:         # if it is binary, change to 0 and 1
        #    le.fit(df1[ccol])
         #   df1[ccol] = le.transform(df1[ccol])

    ## scale numerical columns
    for ncol in numcols:
        df1[f"{ncol}scaled"] = scaler.fit_transform(df1[[ncol]])
        if drop_orig_col==True:
            df1.drop(ncol, axis=1, inplace=True)  # drop the original numerical variable (optional)

    return df1


## A function to perform the logistic regression
## Input: preprocessed pandas dataframe, target variable
## Output: a pandas df with the coefficients and constant value
def run_logreg(prep_df, target):
    X = prep_df.drop(columns=[target])       # separate the predictors
    y = prep_df[target]                      # target 
    X = sm.add_constant(X)                   # need to add the constant
    model = Logit(y, X)                      # run the model
    logit_model = model.fit()
    coefs = logit_model.params               # get the coefficients
    result = pd.DataFrame(zip(X.columns, np.transpose(coefs)), columns=['features', 'coef']).set_index('features').T
    return result


## A function to get the mean squared error between the original coeffs and the synthetic ones
## Input: original coefficients, synthetic dataset, list of categorical variables, list of numerical vars,
## the target variable, list of reference levels. And initialised le and scaler instances
## Output: the mse value
def get_reg_mse(source_coeffs, pop_df, catvars, numvars, target, reflevels, le, scaler):
    ## preprocess the dataframe
    df = preprocess_cols(pop_df, catvars, numvars, le, scaler, drop_orig_col=True)
    ## drop reference levels
    df = df.drop(reflevels, axis=1)
    ## run the logistic regression and get the coefficients
    df_coeffs = run_logreg(df, target)
    ## get the mean squared difference
    diff = (source_coeffs-df_coeffs)**2
    diff = diff.fillna(1)                ## set NAN to 1 - this could be any (big) value
    mean_diff = (diff.mean(axis=1))[0]   ## get the mean 
    return mean_diff


def get_reg_mse_paral(source_coeffs,population,catvars, numvars, target, reflevels, le, scaler,joblib_num_cores=-2,joblib_max_nbytes='10M'):
    scores = Parallel(n_jobs=joblib_num_cores,max_nbytes=joblib_max_nbytes)(delayed(get_reg_mse)(source_coeffs,_,catvars, numvars, target, reflevels, le, scaler) for _ in population)
    return scores


###############################################################################################################




###############################################################################################################
############# TCAP
## Input:  original = original pandas dataframe
##         synth = synthetic pandas dataframe
##         num_keys = number of key variables
##         target = target variable name
##         key1,.. = the key value names. Minimum of to be used 3, maximum of six
## Output: TCAP value (undefined), TCAP zero
## an updated tcap version with tau and eqmax

def tcap_new(original, synth, num_keys, target, key1, key2, key3, key4=None, key5=None, key6=None, 
         tau=1, eqmax=None, verbose=False):
    
    # define the keys and target. using the num_keys parameter means that a dataset with any number of columns can
    # be used, and only the relevant keys analysed
    if num_keys==6:
        keys_target = [key1,key2,key3,key4,key5,key6,target]
    if num_keys==5:
        keys_target = [key1,key2,key3,key4,key5,target]
    if num_keys==4:
        keys_target = [key1,key2,key3,key4,target] 
    if num_keys==3:
        keys_target = [key1,key2,key3,target]
    
    # select just the required columns (keys and target)    
    orig = original[keys_target]
    syn = synth[keys_target]
    
    # set eqmax to the length of source dataset (unless otherwise set)
    if eqmax == None:
        eqmax = len(orig)
    
    # count the categories for the target (for calculating baseline)
    uvd = orig[target].value_counts()
    
    # use groupby to get the equivalance classes for synth data
    eqkt_syn = pd.DataFrame({'count' : syn.groupby( keys_target ).size()}).reset_index()           # with target
    eqk_syn = pd.DataFrame({'count' : syn.groupby( keys_target[:num_keys] ).size()}).reset_index() # without target
    # equivalance classes for orig data without target
    eqk_orig = pd.DataFrame({'count' : orig.groupby( keys_target[:num_keys] ).size()}).reset_index()
    
    if eqmax < len(orig):
        # drop those classes with a count greater than eqmax
        eqk_orig.drop(eqk_orig[eqk_orig['count'] > eqmax].index, inplace=True)
    # merge with orig to calculate eqmaxcount and baseline    
    orig_merge_eqk = pd.merge(orig, eqk_orig, on= keys_target[:num_keys]) 
    orig_merge_eqk.rename({'count': 'count_eqk_orig'}, axis=1, inplace=True)
    eqmaxcount = len(orig_merge_eqk) # the no. of original records with eq class size less than eqmax
    
    # calculate the baseline
    uvt = sum(uvd[orig_merge_eqk[target]]/sum(uvd))
    baseline = uvt/eqmaxcount
    
    if tau == 1: 
        # calculate synthetic cap score. merge syn eq classes (with keys) with syn eq classes (with keys/target)
        temp1 = eqk_syn.merge(eqkt_syn, on=keys_target[:num_keys])
        temp1['prop'] = temp1['count_y']/temp1['count_x']
        # filter out those less than tau
        temp1 = temp1[temp1['prop'] >= tau]
        # merge with original, if in syn eq classes (just keys) then this is a matching record (taub)
        temp1 = temp1.merge(orig_merge_eqk, on=keys_target[:num_keys], how='inner')
        matching_records = len(temp1)
        # drop records where the targets are not equal
        temp1 = temp1[temp1[target + '_x']==temp1[target + '_y']]
        dcaptotal = len(temp1)
    else:
        # if orig record is in the syn equivalence classes (just keys) then this is a matching record (elliot)
        temp = orig_merge_eqk.merge(eqk_syn, on=keys_target[:num_keys], how='inner')
        temp.rename({'count': 'count_eqk_syn'}, axis=1, inplace=True)
        matching_records = len(temp)
        # if orig record (with target and keys) is in syn eq classes (with target and keys)
        temp = temp.merge(eqkt_syn, on=keys_target, how='inner') 
        temp.rename({'count': 'count_eqkt_syn'}, axis=1, inplace=True)
        # calculate the cap score (number in class for keys/target divided by number in class for keys)
        temp['prop'] = temp['count_eqkt_syn']/temp['count_eqk_syn']
        # drop records with prop less than tau
        temp.drop(temp[temp['prop'] < tau].index, inplace=True)    
        dcaptotal = temp['prop'].sum()
        
    if matching_records == 0:
        tcap_undef = 0
    else:
        tcap_undef = dcaptotal/matching_records
        
    tcap_zero = dcaptotal/eqmaxcount
    
    # this is [the TCAP with non-matches undefined, with non-matches as zero, and the baseline]
    #output = ([tcap_undef,tcap_zero,baseline])
    output = ([tcap_undef,tcap_zero])
    
    if verbose==True:
        print('\nTCAP calculation')
        print('===============')
        print('The total number of records in the source dataset is: ', len(orig))
        print('The total number of records in the target dataset is: ', len(syn))
        print('The target variable is: ', target)
        print('The key size is: ', num_keys)
        print('The keys are: ', key1, key2, key3, key4, key5, key6)
        print('Tau is set to: ', tau)
        print('Number of matching records: ', matching_records)
        print('DCAP total is: ', dcaptotal)
        print('eqmax count is: ', eqmaxcount)
        print('Maximum source equivalence class size is set to: ', eqmax)
        print('TCAP with non-matches undefined is: ', tcap_undef)
        print('TCAP with non-matches as zero is: ', tcap_zero)
        print('The baseline is: ', baseline)

    return(output)


## A function to return all possible key combinations of 3 - 6 (or more) keys, given a list of keys
## Returns a list of lists
def get_all_key_combos(key_list):
    number_keys = len(key_list)
    #if number_keys > 6 :     # only use up to 6 keys for TCAP
    #    number_keys=6
    combo_list = []
    if number_keys > 2:      # if number of keys is greater than 3
        for i in range(3,number_keys+1): 
            combos = sorted(set(itertools.combinations(key_list, i)))
            combos = [list(j) for j in combos]
            combo_list.extend(combos)
    return combo_list
