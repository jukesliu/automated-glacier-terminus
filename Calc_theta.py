#!/usr/bin/env python
# coding: utf-8

# In[7]:


#!/usr/bin/python

if True==True:
    import pandas as pd
    import numpy as np
    import sys
    # read in the manual terminus position
    manual_df = pd.read_csv('/media/jukes/jukes1/Manual/manual_tpos_c1.csv', dtype=str,sep=',')
    
    #SPLIT INTO 3 DATAFRAMES FOR 3 FLOWLINES:
    auto_path = '/home/jukes/Documents/Sample_glaciers/'
    manual50 = manual_df[['BoxID','datetimes', 'intersect_x', 'intersect_y', 
                                          'tpos50']].copy().reset_index(drop=True).rename(columns={"tpos50": "tpos"})
    manual25 = manual_df[['BoxID','datetimes', 'intersect_x', 'intersect_y', 
                                          'tpos25']].copy().reset_index(drop=True).rename(columns={"tpos25": "tpos"})
    manual75 = manual_df[['BoxID','datetimes', 'intersect_x', 'intersect_y',
                                          'tpos75']].copy().reset_index(drop=True).rename(columns={"tpos75": "tpos"})
    thetas = []
    #FOR EACH GLACIER BOXID:
    BoxIDs = list(set(manual_df.BoxID))
    for BoxID in BoxIDs:
        #grab automated tpos
        auto50 = pd.read_csv(auto_path+'Tpos_Box'+BoxID+'_flowline50_filtered.csv', dtype=str,sep=',')
        auto25 = pd.read_csv(auto_path+'Tpos_Box'+BoxID+'_flowline25_filtered.csv', dtype=str,sep=',')
        auto75 = pd.read_csv(auto_path+'Tpos_Box'+BoxID+'_flowline75_filtered.csv', dtype=str,sep=',')
        autodfs = [auto50, auto25, auto75]
        #grab manual tpos that corresponds to just boxID
        manual50_df = manual50[manual50.BoxID == BoxID].copy()
        manual25_df = manual25[manual25.BoxID == BoxID].copy()
        manual75_df = manual75[manual75.BoxID == BoxID].copy()
        manualdfs = [manual50_df, manual25_df, manual75_df]
        #calculate difference in terminus positions along the three flowlines
        lists3 = []; lists3_norm = []
        for i in range(0, len(manualdfs)):
            man = manualdfs[i]; auto = autodfs[i]; # sigma = sigmas[i]
            compare_df = man.merge(auto, how='inner', on=['datetimes'])
            #cast terminus positions into float values
            compare_df = compare_df.astype({'tpos_x': 'float', 'tpos_y': 'float'})
            #subtract the absolute value of the difference and put into df as a column named "diff"
            compare_df['diff'] = abs(np.array(compare_df.tpos_x) - np.array(compare_df.tpos_y))  
            lists3.append(list(compare_df['diff']))  
        diff_all = lists3[0]+lists3[1]+lists3[2] #list of all the differences between manual and auto
    #     normalizeddiff_all = lists3_norm[0]+lists3_norm[1]+lists3_norm[2] #list of all the normalized differences

        N = len(diff_all) #number of total intersections

        #CALCULATE THETA:
        theta = (1.0/N)*(np.nansum(diff_all)) #sum of differences normalized by average sigma
        thetas.append(theta)
                
    #CALCULATE OVERALL THETA
    theta_all = np.nanmean(thetas)
    print(theta_all)
    
    import sys
    sys.exit()


# In[ ]:




