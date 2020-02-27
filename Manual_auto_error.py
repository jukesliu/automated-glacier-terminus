#!/usr/bin/env python
# coding: utf-8

# ## Code to calculate the auto and manual difference using an objective function
# 
# Weighted least squares solution

# ### Import packages, functions, manual and automated data

# In[65]:


import pandas as pd
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import numpy.ma as ma
import datetime
import math

manual_path = '/media/jukes/jukes1/Manual/'; manual_filename = 'manual_tpos.csv'
auto_path = '/home/jukes/Documents/Sample_glaciers/'


# In[68]:


os.chdir('/home/jukes/automated-glacier-terminus') #import necessary functions:
from automated_terminus_functions import distance


# In[73]:


coord1 = [624431.8665895949, 8576051.819328722]
coord2 = [624337.7591887069, 8576049.807048056]


# In[74]:


distance(coord1[0], coord1[1], coord2[0], coord2[1])


# In[76]:


#MANUAL TERMINUS POSITIONS
manual_df = pd.read_csv(manual_path+manual_filename, dtype=str,sep=',')

#SPLIT INTO 3 DATAFRAMES FOR 3 FLOWLINES:
manual50 = manual_df[['BoxID','datetimes', 'intersect_x', 'intersect_y', 
                                      'tpos50']].copy().reset_index(drop=True).rename(columns={"tpos50": "tpos"})
manual25 = manual_df[['BoxID','datetimes', 'intersect_x', 'intersect_y', 
                                      'tpos25']].copy().reset_index(drop=True).rename(columns={"tpos25": "tpos"})
manual75 = manual_df[['BoxID','datetimes', 'intersect_x', 'intersect_y',
                                      'tpos75']].copy().reset_index(drop=True).rename(columns={"tpos75": "tpos"})


# In[46]:


#SIGMAS (DATA ERRORS) ALONG EACH FLOWLINE (FROM INTERANALYST DIFFERENCES)
sigmas = [35.02, 27.65, 30.45]
sigma_avg = np.average(sigmas); print(sigma_avg)


# In[56]:


theta1s = []; theta2s = []
#FOR EACH GLACIER BOXID:
BoxIDs = list(set(manual_df.BoxID))
for BoxID in BoxIDs:
    print("Box"+BoxID)
    #grab automated tpos
    auto50 = pd.read_csv(auto_path+'Tpos_Box'+BoxID+'_flowline50_filtered.csv', dtype=str,sep=',')
    auto25 = pd.read_csv(auto_path+'Tpos_Box'+BoxID+'_flowline25_filtered.csv', dtype=str,sep=',')
    auto75 = pd.read_csv(auto_path+'Tpos_Box'+BoxID+'_flowline75_filtered.csv', dtype=str,sep=',')
    autodfs = [auto50, auto25, auto75]
    #grab manual tpos that corresponds to just boxID
    manual50_df = manual50[manual50.BoxID == BoxID].copy()
    manual25_df = manual25[manual25.BoxID == BoxID].copy()
    manual75_df = manual75[manual75.BoxID == BoxID].copy()
    manualdfs = [manual50, manual25, manual75]
    #calculate difference in terminus positions along the three flowlines
    lists3 = []; lists3_norm = []
    for i in range(0, len(manualdfs)):
        man = manualdfs[i]; auto = autodfs[i]; sigma = sigmas[i]
        compare_df = man.merge(auto, how='inner', on=['datetimes'])
        #cast terminus positions into float values
        compare_df = compare_df.astype({'tpos_x': 'float', 'tpos_y': 'float'})
        #subtract the absolute value of the difference and put into df as a column named "diff"
        compare_df['diff'] = abs(np.array(compare_df.tpos_x) - np.array(compare_df.tpos_y))  
        compare_df['diff/sigma'] = abs(np.array(compare_df.tpos_x) - np.array(compare_df.tpos_y))/sigma
        lists3.append(list(compare_df['diff']))  
        lists3_norm.append(list(compare_df['diff/sigma']))
    diff_all = lists3[0]+lists3[1]+lists3[2] #list of all the differences between manual and auto
    normalizeddiff_all = lists3_norm[0]+lists3_norm[1]+lists3_norm[2] #list of all the normalized differences
    N = len(diff_all) #number of total intersections
    
    #CALCULATE THETA:
    theta1 = (1.0/N)*np.sum(normalizeddiff_all) #sum of normalized differences along flowlines
    theta2 = (1.0/N)*(np.sum(diff_all)/sigma_avg) #sum of differences normalized by average sigma
    theta1s.append(theta1); theta2s.append(theta2)
    print("Theta values:",theta1, theta2)


# In[60]:


list(zip(columns, theta1_for_df, theta2_for_df))


# In[67]:


#CALCULATE OVERALL THETA and write results to csv
theta1_all = np.average(theta1s)
theta2_all = np.average(theta2s)

#organize data
columns = ['Theta_avg']+BoxIDs
theta1_for_df = [theta1_all]+theta1s
theta2_for_df = [theta2_all]+theta2s
#write to csv
pd.DataFrame(list(zip(columns, theta1_for_df, theta2_for_df)), 
             columns=['ID', 'theta1', 'theta2']).to_csv(manual_path+'thetas.csv', sep=',') 

#ADJUST FILENAME TO INCLUDE PARAMETERS OR SOMETHING


# In[80]:


def objective_func(manual_df):
    #SPLIT INTO 3 DATAFRAMES FOR 3 FLOWLINES:
    manual50 = manual_df[['BoxID','datetimes', 'intersect_x', 'intersect_y', 
                                          'tpos50']].copy().reset_index(drop=True).rename(columns={"tpos50": "tpos"})
    manual25 = manual_df[['BoxID','datetimes', 'intersect_x', 'intersect_y', 
                                          'tpos25']].copy().reset_index(drop=True).rename(columns={"tpos25": "tpos"})
    manual75 = manual_df[['BoxID','datetimes', 'intersect_x', 'intersect_y',
                                          'tpos75']].copy().reset_index(drop=True).rename(columns={"tpos75": "tpos"})
    #SIGMAS (DATA ERRORS) ALONG EACH FLOWLINE (FROM INTERANALYST DIFFERENCES)
    sigmas = [35.02, 27.65, 30.45]; sigma_avg = np.average(sigmas);
    
    theta1s = []; theta2s = []
    #FOR EACH GLACIER BOXID:
    BoxIDs = list(set(manual_df.BoxID))
    for BoxID in BoxIDs:
        print("Box"+BoxID)
        #grab automated tpos
        auto50 = pd.read_csv(auto_path+'Tpos_Box'+BoxID+'_flowline50_filtered.csv', dtype=str,sep=',')
        auto25 = pd.read_csv(auto_path+'Tpos_Box'+BoxID+'_flowline25_filtered.csv', dtype=str,sep=',')
        auto75 = pd.read_csv(auto_path+'Tpos_Box'+BoxID+'_flowline75_filtered.csv', dtype=str,sep=',')
        autodfs = [auto50, auto25, auto75]
        #grab manual tpos that corresponds to just boxID
        manual50_df = manual50[manual50.BoxID == BoxID].copy()
        manual25_df = manual25[manual25.BoxID == BoxID].copy()
        manual75_df = manual75[manual75.BoxID == BoxID].copy()
        manualdfs = [manual50, manual25, manual75]
        #calculate difference in terminus positions along the three flowlines
        lists3 = []; lists3_norm = []
        for i in range(0, len(manualdfs)):
            man = manualdfs[i]; auto = autodfs[i]; sigma = sigmas[i]
            compare_df = man.merge(auto, how='inner', on=['datetimes'])
            #cast terminus positions into float values
            compare_df = compare_df.astype({'tpos_x': 'float', 'tpos_y': 'float'})
            #subtract the absolute value of the difference and put into df as a column named "diff"
            compare_df['diff'] = abs(np.array(compare_df.tpos_x) - np.array(compare_df.tpos_y))  
            compare_df['diff/sigma'] = abs(np.array(compare_df.tpos_x) - np.array(compare_df.tpos_y))/sigma
            lists3.append(list(compare_df['diff']))  
            lists3_norm.append(list(compare_df['diff/sigma']))
        diff_all = lists3[0]+lists3[1]+lists3[2] #list of all the differences between manual and auto
        normalizeddiff_all = lists3_norm[0]+lists3_norm[1]+lists3_norm[2] #list of all the normalized differences
        N = len(diff_all) #number of total intersections

        #CALCULATE THETA:
        theta1 = (1.0/N)*np.sum(normalizeddiff_all) #sum of normalized differences along flowlines
        theta2 = (1.0/N)*(np.sum(diff_all)/sigma_avg) #sum of differences normalized by average sigma
        theta1s.append(theta1); theta2s.append(theta2)
        #print("Theta values:",theta1, theta2)   
        
    #CALCULATE OVERALL THETA
    theta1_all = np.average(theta1s); theta2_all = np.average(theta2s)
    #organize data in dataframe
    column_titles = ['Theta_avg']+BoxIDs
    theta1_for_df = [theta1_all]+theta1s; theta2_for_df = [theta2_all]+theta2s
    #write to csv
    theta_df = pd.DataFrame(list(zip(column_titles, theta1_for_df, theta2_for_df)), 
                 columns=['ID', 'theta1', 'theta2'])
    return theta_df 


# In[78]:


# objective_func(manual_df)


# In[ ]:




