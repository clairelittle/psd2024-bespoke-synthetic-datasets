## October 2023
## Functions for the EA for the teaching dataset where we would only have the distribution of the
## data and some regression coefficients and tables
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
from operator import add
from collections import Counter
import EA_utility_risk_functions as urfuncs



###############################################################################################################
## This will replace missing NA values (dependent on data type) in a pandas dataframe. 
## Input dataframe, returns dataframe. 
## Modify as required - Missing categorical are replaced with 'blank', numeric with -99
def replace_missing(dataset):
    # get a dictionary of the different data types
    types = dataset.dtypes.to_dict()
    # replace object or categorical NAs with 'blank', and numerical with -99
    for col_nam, typ in types.items():
        if (typ == 'O' or typ == 'c'):
            dataset[col_nam] = dataset[col_nam].fillna('blank')
        if (typ == 'float64' or typ == 'int64'):
            dataset[col_nam] = dataset[col_nam].fillna(-99)
    return(dataset)


###############################################################################################################
## Generate a "random" pandas dataframe the same size and shape as the original dataframe. Assumes NA values
## have been replaced with a suitable value (e.g. -999)
## By default the type is "univariate" - this will sample from the univariate distribution of each variable. 
## If the type is set to "uniform", it samples from the uniform distribution.
## If the type is set to "noise", it samples from uniform distribution of 100 values (0-99, can edit this)
## That is, the "noise" values are not taken from the source data (although some may match) but just made up
## Input pandas dataframe, output pandas dataframe
def generate_random_dataset(source_dataset, type="univariate", random_seed=None):
    ## Extract information from the original/source dataset
    num_variables = source_dataset.shape[1]          # number of variables
    num_rows = source_dataset.shape[0]               # number of records
    var_names = list(source_dataset.columns.values)  # variable names
    # Create a new (empty) dataframe using the column names
    random_df = pd.DataFrame(columns=var_names)
    # Use a for loop to populate the columns with data. This loops through the variables/columns and
    # gets the number of unique categories and their values and then samples from the univariate or uniform 
    # distribution. It only samples values that are in the original data
    for i in range(num_variables):
        if random_seed==None:                                 # If random seed is not specified then
            random.seed()                                     # set a random seed (defaults to current system time)
        else:                                                 # otherwise use the specified random seed but add i to it so it is
            random.seed(random_seed+i)                        # different for each dataset generated (but still reproducible)
        if type=='univariate':                                # use actual proportions for univariate data
            random_df[var_names[i]] = random.choices(list(source_dataset[var_names[i]]),k=num_rows)
        elif type=='uniform':                                 # use even proportions for uniform data
            weight = len(list(source_dataset[var_names[i]].value_counts().index))
            random_df[var_names[i]] = random.choices(list(source_dataset[var_names[i]].value_counts().index),
                                                     weights=[100/weight]*weight,k=num_rows)
        elif type=='noise':                                   
            vals = list(range(100))                           # if noise, use list of 100 values
            val_weights = len(vals) * [1/len(vals)]           # with equal proportions
            random_df[var_names[i]] = random.choices(vals, weights=val_weights, k=num_rows)
        else:
            raise Exception('type must be "uniform", "univariate" or "noise"')
    
    return(random_df)

#### generate many datasets
def generate_multiple_random_datasets(source_dataset, num_datasets=1, type="univariate", random_seed=None):
    dataset_list = []
    if random_seed==None:                           # If random seed is not specified then
        random_seed=99                              # set a random seed
    for i in range(num_datasets):
        ran_seed = random_seed+i                      # use different random seed for each dataset
        dataset = generate_random_dataset(source_dataset, type, ran_seed)
        dataset_list.append(dataset)
    return(dataset_list)


##### FL version, this takes the dataset distribution (instead of original dataset) as input (generated by the
## get_variable_dn_fl() function). And returns a uniform, univariate or noise dataset
def generate_random_dataset_fl(source_dns, dist_type="univariate", random_seed=None):
    ## Extract information from the source distribution
    num_variables = len(source_dns[3])               # number of variables
    num_rows = source_dns[4]                         # number of records
    var_names = source_dns[3]                        # variable names
    # Create a new (empty) dataframe using the column names
    random_df = pd.DataFrame(columns=var_names)
    # Use a for loop to populate the columns with data. This loops through the variables/columns and
    # gets the number of unique categories and their values and then samples from the univariate or uniform 
    # distribution. It only samples values that are in the original data
    for i in range(num_variables):           
        if random_seed==None:                             # If random seed is not specified then
            random.seed()                                 # set a random seed (defaults to current system time)
        else:                                             # otherwise use the specified random seed but add i to it so it is
            random.seed(random_seed+i)                    # different for each dataset generated (but still reproducible)
        if dist_type=='univariate':                            # use actual proportions for univariate data
            random_df[var_names[i]] = random.choices(source_dns[0][i], weights = source_dns[1][i], k=num_rows)
        elif dist_type=='uniform':                             # use even proportions for uniform data
            random_df[var_names[i]] = random.choices(source_dns[0][i], weights = source_dns[2][i], k=num_rows)
        elif dist_type=='noise':                                   
            vals = list(range(100))                       # if noise, use list of 100 values
            val_weights = len(vals) * [1/len(vals)]       # with equal proportions
            random_df[var_names[i]] = random.choices(vals, weights=val_weights, k=num_rows)
        else:
            raise Exception('type must be "uniform", "univariate" or "noise"')
    
    return(random_df)

#### generate many datasets FL version, slight modification
def generate_multiple_random_datasets_fl(source_dns, num_datasets=1, dist_type="univariate", random_seed=None):
    dataset_list = []
    if random_seed==None:                           # If random seed is not specified then
        random_seed=99                              # set a random seed
    for i in range(num_datasets):
        ran_seed = random_seed+i                    # use different random seed for each dataset
        dataset = generate_random_dataset_fl(source_dns, dist_type, ran_seed)
        dataset_list.append(dataset)
    return(dataset_list)



###############################################################################################################
## Get distribution of pandas dataset
## this gets the values and proportions for each variable (column)
## Input pandas dataframe, output list with unique values and proportions, and uniform distribution (unif_dist)
def get_variable_dn(source_dataset):
    var_names = list(source_dataset.columns.values)       # variable names
    num_vars = source_dataset.shape[1]                    # number of variables
    num_rows = source_dataset.shape[0]                    # number of records
    values, dist, unif_dist = [], [], []                  # create empty lists to store results in
    for i in range(num_vars):
        v = list(source_dataset[var_names[i]].value_counts().index) # get category values
        values.append(v)
        d = list(source_dataset[var_names[i]].value_counts()) # get number for each category
        d = [x/num_rows for x in d]                           # get proportions by dividing by total no. rows
        dist.append(d)
        u = len(v) * [1/len(v)]                               # use number of categories to get uniform dist
        unif_dist.append(u)
    return [values, dist, unif_dist]


## Get distribution of pandas dataset (same as above, but returns number of values and records as well)
## this gets the values and proportions for each variable (column)
## Input pandas dataframe, output list with unique values and proportions, and uniform distribution (unif_dist)
def get_variable_dn_fl(source_dataset):
    var_names = list(source_dataset.columns.values)       # variable names
    num_vars = source_dataset.shape[1]                    # number of variables
    num_rows = source_dataset.shape[0]                    # number of records
    values, num_values, dist, unif_dist = [], [], [], []  # create empty lists to store results in
    for i in range(num_vars):
        v = list(source_dataset[var_names[i]].value_counts().index) # get category values
        values.append(v)
        d = list(source_dataset[var_names[i]].value_counts()) # get number for each category
        num_values.append(d)
        e = [x/num_rows for x in d]                           # get proportions by dividing by total no. rows
        dist.append(e)
        u = len(v) * [1/len(v)]                               # use number of categories to get uniform dist
        unif_dist.append(u)
    return [values, dist, unif_dist, #num_values, 
            var_names, num_rows]




################################################################################################################
################################################################################################################
## TOURNAMENT SELECTION
## this randomly chooses two datasets, selects the one with the highest sim score, then returns the index. 
## sim_vals is a list of similarity scores, k can be set to any number but defaults to 2.
## If the scores are all the same, then one will be chosen randomly
def selection(sim_vals, k=2):    
    if len(set(sim_vals)) < 2: # if scores are all the same then choose randomly (not just the first index)
        return randint(0,len(sim_vals))
    candidates = random.sample(list(enumerate(sim_vals)), k) # sample k candidates (returns index and value)
    max_tuple = max(candidates, key=lambda x:x[1])           # get the maximum value
    max_tuple_index = candidates.index(max_tuple)            # get the index of this
    index = candidates[max_tuple_index][0]    
    return index


## TOURNAMENT SELECTION - MINIMUM
## this randomly chooses two datasets, selects the one with the lowest score, then returns the index. 
## sim_vals is a list of scores, k can be set to any number but defaults to 2.
## If the scores are all the same, then one will be chosen randomly
def selection_min(sim_vals, k=2):    
    if len(set(sim_vals)) < 2: # if scores are all the same then choose randomly (not just the first index)
        return randint(0,len(sim_vals))
    candidates = random.sample(list(enumerate(sim_vals)), k) # sample k candidates (returns index and value)
    max_tuple = min(candidates, key=lambda x:x[1])           # get the minimum value
    max_tuple_index = candidates.index(max_tuple)            # get the index of this
    index = candidates[max_tuple_index][0]    
    return index
################################################################################################################
################################################################################################################




################################################################################################################
################################################################################################################
## CROSSOVER
## crossover two parents (pandas dataframes) to create two children (pandas dataframes)
## this crossover is a random selection of records from each (not a block)
## cross_rate - whether to crossover - is set to 0.7 by default, which seems fairly standard
## cross_prob is probability of crossover for each record, this is 0.1 by default but can be set
def crossover(p1, p2, cross_rate=None, cross_prob=None):
    c1, c2 = p1.copy(), p2.copy()                                  # children are copies of parents by default
    if cross_rate==0:                                              # if crossover rate is 0, don't do crossover
        return [c1,c2]
    if cross_rate==None or cross_rate > 1 or cross_rate < 0:       # if cross_rate not specified or invalid set to 0.7
        cross_rate = 0.8
    if rand() <= cross_rate:                                       # check for whether to crossover
        if cross_prob == None or cross_prob > 1 or cross_prob < 0: # if cross_prob not specified or invalid set to 0.1
            cross_prob = 0.1 # rand()                              # (or could set it to a random value between 0 & 1)

        idx_list = []
        for rec in range(len(p1)):                                 # for each record in the dataset
            if rand() <= cross_prob:                               # decide whether to crossover
                idx_list.append(rec)                               # store the list of indices  
              
        # perform crossover, taking idx_list rows from p1 and !idx_list rows from p2 (and the opposite for c2)
        c1 = pd.concat([p1.loc[p1.index.isin(idx_list)], p2.loc[~ p2.index.isin(idx_list)]], ignore_index=True)
        c2 = pd.concat([p2.loc[p2.index.isin(idx_list)], p1.loc[~ p1.index.isin(idx_list)]], ignore_index=True)
        # note this reorders the rows
    return [c1, c2]

################################################################################################################
################################################################################################################



################################################################################################################
################################################################################################################
## MUTATION 
## input pandas dataframe, the source data distributions, mutation rate (defaults to 1/number of vars)
## and which distribution to use for mutation (default is univariate, but can do uniform too)
## Added later - if there is a column with weights (it would be the last one), ignore it
## Output is mutated dataframe
## Could be coded in a more numpy/pandas way, but takes only a few seconds to run as is
def mutation(dataset, distributions, mut_rate=None, mut_dist='univariate',weights=False):
    if weights==True:
        num_vars = dataset.shape[1] - 1 # exclude the last variable (Weight)
    else:
        num_vars = dataset.shape[1]     # get the number of variables
    if mut_rate==None:                  # if the mut_rate is not specified set it as 1/number of variables
        mut_rate = 1/num_vars
    for rec in range(len(dataset)):     # for each record in the dataset
        for j in range(num_vars):       # for each variable in the record
            if rand() <= mut_rate:      # check for a mutation
                if mut_dist=='uniform': # if uniform is specified use the uniform distribution
                    dataset.iloc[rec,j] = random.choices(distributions[0][j], weights = distributions[2][j], k=1)[0]
                else:                   # if not specified or anything else use the univariate distribution
                    dataset.iloc[rec,j] = random.choices(distributions[0][j], weights = distributions[1][j], k=1)[0]
    return dataset



## Mutation where the variables to be mutated are selected
## Same as above, but the "variables_select" parameter contains a list of variables to mutate (all are mutated
## if not specified)
def mutation_select(dataset, distributions, mut_rate=None, mut_dist='univariate',weights=False, variables_select=None):
    ## If there are weights, exclude them (this assumes weights are in the final column)
    if weights==True:
        num_vars = dataset.shape[1] - 1   # exclude the last variable (Weight)
    else:
        num_vars = dataset.shape[1]       # get the number of variables
    if mut_rate==None:                    # if the mut_rate is not specified set it as 0.005
        mut_rate = 0.005
    ## Select which variables to use
    if variables_select is not None:      # if there is a list of variables to mutate on, use those
        vars_to_mutate = variables_select
    else:                                 # if there is no list, use them all (excluding weights if included)
        vars_to_mutate = list(dataset.columns[:num_vars])
    ## get the column index of the variables to mutate on
    idx_list = []
    for var in vars_to_mutate:
        idx_list.append(dataset.columns.get_loc(var))
    ## Then mutate by cycling through each record and variable (there are more efficient ways than this)
    for rec in range(len(dataset)):     # for each record in the dataset
        for j in idx_list:              # for each variable that we are mutating
            if rand() <= mut_rate:      # check for a mutation
                #print(f"{j} and record {rec}")
                if mut_dist=='uniform': # if uniform is specified use the uniform distribution
                    dataset.iloc[rec,j] = random.choices(distributions[0][j], weights = distributions[2][j], k=1)[0]
                else:                   # if not specified or anything else use the univariate distribution
                    dataset.iloc[rec,j] = random.choices(distributions[0][j], weights = distributions[1][j], k=1)[0]
    return dataset
################################################################################################################
################################################################################################################



################################################################################################################
################################################################################################################
## SELECTION OF NEXT GENERATION
## If not using the children as the next generation, put parents and children together and choose the strongest
## based on the weighted sum score.
## (In the case of all weighted sums being the same a random selection will be chosen).
## Input: population similarity scores, child similarity scores, the population and the children (list of DFs)
## Output: population and their similarity score

### where we are minimising the weighted score, and there are multiple scores. This gets the lowest scores
def next_gen_min_v2(pop_scores, child_scores, population, children):
    popn_size = len(population)                                    # get the population size
    population = population + children                             # combine parent & child populations
    com0 = pop_scores[0] + child_scores[0]                         # combine parent & child  scores
    com1 = pop_scores[1] + child_scores[1]
    com2 = pop_scores[2] + child_scores[2]
    avg_com = pop_scores[3] + child_scores[3]                     
    
    if len(set(avg_com)) < 2: 
        indx = np.random.permutation(popn_size*2)[:popn_size]      # if scores are all the same, choose randomly
    else:
        indx = list((np.array(avg_com)).argsort()[:popn_size])     # get index of lowest scores
        
    population = list(population[i] for i in indx)                 # keep the top datasets
    com0 = list(com0[i] for i in indx)                             # keep the corresponding scores
    com1 = list(com1[i] for i in indx)                             # keep the corresponding scores
    com2 = list(com2[i] for i in indx)                             # keep the corresponding scores
    avg_com = list(avg_com[i] for i in indx)                       # keep the corresponding weighted sum scores
    # keep as a list of lists
    scores_com =[]
    scores_com.append(com0)
    scores_com.append(com1)
    scores_com.append(com2)
    scores_com.append(avg_com)   
    return population, scores_com # return best population with their scores



################################################################################################################
################################################################################################################



################################################################################################################
################################################################################################################
##### CREATE STARTING POINT
## Creates the initial population and calculates the risk/utility and weighted sum scores
## input:  source_distribution = distribution of original data 
##         pop_size = number of dataframes in the population
##         initial_pop_dn = initial population distribution (uniform,univariate)
##         rand_seed = random seed for generating the initial population (set automatically if not specified)
##         filename = name of file to store scores in
##         joblib_num_cores = number of cores to use in parallel processing (-2 is all but one)
##         joblib_max_nbytes = this threw an error at '1M' so it is set larger ('10M')
##         verbose = will save the scores
## output: initial_pop = population of synthetic dataframes (uniform or univariate distribution)
##         scores_all = list of mean squared error scores for means of the source data compared to synth means
##                      A file with the scores for each generation


def setup_initial_pop(source_distribution, pop_size, initial_pop_dn,rand_seed,filename,
                      output1,output2,output3,output_wts,catvars, numvars, target, reflevels, 
                      le, scaler, var_name1,var_name2, 
                      joblib_num_cores=-2,joblib_max_nbytes='10M', verbose=True):
    
    ##### generate datasets (using uniform distribution as default) to create initial population
    initial_pop = generate_multiple_random_datasets_fl(source_distribution, num_datasets=pop_size, 
                                                       dist_type=initial_pop_dn, random_seed=rand_seed) 
    
    if verbose==True:
        print(f'Start EA:\nPopulation initialised using {initial_pop_dn} distribution') 
    
    ##### Get the scores
    ## compare the data to the outputs
    scores = get_all_scores(initial_pop,output1,output2,output3,catvars, numvars, target, 
                            reflevels, le, scaler,var_name1,var_name2,
                            jlib_num_cores=joblib_num_cores,jlib_max_nbytes=joblib_max_nbytes)

    if verbose==True:
        print(f'Scores calculated') 

    ## Get the average weighted score
    avg_score = list(map(add, [i * output_wts[0] for i in scores[0]],
                         (map(add, [i * output_wts[1] for i in scores[1]],
                              [i * output_wts[2] for i in scores[2]])) )) 

    scores.extend([avg_score])
    write_scores(avg_score, filename)
    
    return initial_pop, scores



######## If we already have a population that we want to start with:
def setup_existing_pop(existing_pop, filename,output1,output2,output3, output_wts,
                       catvars, numvars, target, reflevels, le, scaler, var_name1,var_name2,
                       joblib_num_cores=-2,joblib_max_nbytes='10M', verbose=True):

    ##### Get the scores
    scores = get_all_scores(existing_pop,output1,output2,output3,catvars, numvars, target, 
                            reflevels, le, scaler,var_name1,var_name2,
                            jlib_num_cores=joblib_num_cores,jlib_max_nbytes=joblib_max_nbytes)

    if verbose==True:
        print(f'Start EA:\nExisting population initialised') 

    ## Get the average weighted score
    avg_score = list(map(add, [i * output_wts[0] for i in scores[0]],
                         (map(add, [i * output_wts[1] for i in scores[1]],
                               [i * output_wts[2] for i in scores[2]])) ))

    scores.extend([avg_score])
    write_scores(avg_score, filename)
    
    return existing_pop, scores




#################### A function to write the scores to a csv file
def write_scores(scores,filename):
    with open(filename, "a", newline='') as f:
        wr = csv.writer(f, dialect='excel')
        wr.writerow(scores)  

def write_scores_multi(scores,filename):
    with open(filename, "a", newline='') as f:
        wr = csv.writer(f, dialect='excel')
        for j in range(len(scores)):
                wr.writerow(scores[j]) 



####################
## calculates all the similarity scores and returns result
def get_all_scores(population,output1,output2,output3,catvars, numvars, target, reflevels, le, scaler,
                   var_name1,var_name2, jlib_num_cores=-2,jlib_max_nbytes='10M'):

    score1 = urfuncs.get_reg_mse_paral(output1,population,catvars, numvars, target, reflevels, le, scaler,jlib_num_cores,jlib_max_nbytes)
    score2 = urfuncs.get_twoway_mse_paral(output2,population,target,var_name1,jlib_num_cores,jlib_max_nbytes)
    score3 = urfuncs.get_twoway_mse_paral(output3,population,target,var_name2,jlib_num_cores,jlib_max_nbytes)

    scores = []
    scores.extend((score1,score2,score3))
    
    return scores




################################################################################################################
################################################################################################################
## Main function, uses parallel processing
## input:  pop = population of dataframes
##         scores_pop = the similarity scores between pop dataframes and source dataset
##         pop_size = number of dataframes in the population
##         source_distribution = distribution of original data (generated by get_variable_dn_fl() function)
##         source_coeffs = the coefficients for the regression
##         cat_cols = list of the categorical columns in the synthetic dataframe
##         num_cols = list of the numerical columns in the synthetic dataframe
##         target = the target variable for the regression
##         labelenc = the initiliazed label encoder for one-hot encoding
##         scal = the initialized scaler for scaling the numerical variables
##         filename = filename to write the scores for each generation to
##         filename_ch = filename for the children scores at each generation
##         joblib_num_cores = for parallel processing (default of -2 uses all but one)
##         joblib_max_nbytes = joblib Parallel.py -threw an error set at '1M' - a larger value is needed (default '10M')
##         cross_rate = crossover rate (default of None uses a value of 0.8)
##         cross_prob = the probability of crossing over a record (default is 0.1)
##         mut_rate = mutation rate (default of None uses 1/number of vars)
##         mut_dist = distribution of mutation values (univariate by default, but could use uniform)
##         mut_vars = list of variables to mutate (if None, all are mutated)
##         next_gen = what to use as the next generation, defaults to children replacing parents
##         verbose = whether to save the scores into a csv file
## output: pop = population of dataframes
##         scores = the risk and utility scores between pop dataframes and source datase


#####################
def GA_parallel(pop, scores_pop, pop_size, source_distribution,filename=None,filename_ch=None,
                output1=None,output2=None,output3=None, output_wts=None,catvars=None, numvars=None, 
                target=None, reflevels=None, le=None, scaler=None, var_name1=None,var_name2=None,
                joblib_num_cores=-2, joblib_max_nbytes='10M',
                cross_rate=None, cross_prob=None, mut_rate=None, mut_dist='univariate',
                next_gen='children',verbose=True):
    
    ##### using pop and their scores, create children
    children = list()
    for _ in range(pop_size//2):
        # get selected parents in pairs, using tournament selection (minimum)
        p1_id, p2_id = selection_min(scores_pop[3]), selection_min(scores_pop[3])
        while p1_id == p2_id: # if p2_id is the same as p1_id, choose again
            p2_id = selection_min(scores_pop[3])
        # do crossover
        child1, child2 = crossover(pop[p1_id],pop[p2_id],cross_rate,cross_prob)
        # store
        children.append(child1)
        children.append(child2)

    ##### mutate first, then apply the weights (if required)
    children = Parallel(n_jobs=joblib_num_cores,max_nbytes=joblib_max_nbytes)(delayed(mutation)(_,source_distribution,mut_rate,mut_dist) for _ in children)

    ##### Get scores for children
    scores_ch = get_all_scores(children,output1,output2,output3,catvars, numvars, target, reflevels, 
                               le, scaler, var_name1,var_name2,
                               joblib_num_cores,joblib_max_nbytes)
    
    ## Get the average weighted score
    avg_score_ch = list(map(add, [i * output_wts[0] for i in scores_ch[0]],
                            (map(add, [i * output_wts[1] for i in scores_ch[1]],
                                 [i * output_wts[2] for i in scores_ch[2]])) ))
    scores_ch.extend([avg_score_ch])
    write_scores(avg_score_ch, filename_ch)


    ##### Either use children as next gen, or combine parents and children and choose the best for next generation/output
    if next_gen == 'children':
        pop = children
        scores = scores_ch
    else:
        pop, scores = next_gen_min_v2(scores_pop, scores_ch, pop, children)

    ##### Store the scores for the chosen generation
    if verbose==True:
        write_scores_multi(scores, filename)

    return pop, scores



