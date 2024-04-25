#!/usr/bin/env python
# coding: utf-8

# # Functions written for automated-glacier-terminus detection
# Jukes Liu. _Github: julialiu18_
# 

# In[10]:


#define a function to calculate terminus change rate:
def calc_changerates3(df):
    import pandas as pd; import numpy as np
    tchange50 = []; tchange25 = []; tchange75 = []
    
    for i in range(0, len(df.index)):
        date = list(df['datetimes'])[i]
        #grab the earliest date
        earliestdate = list(df['datetimes'])[0]
        
        flowlines = ['tpos50', 'tpos25', 'tpos75']
        flowline_changes = []
        for flowline in flowlines:
            tpos = list(df[flowline])[i]
            t_
            if date==earliestdate:
                changerate = np.NaN
            else:
                #grab all other subsequent entries
                t = date; x=tpos; counter=1; 
                t_prev = list(df['datetimes'])[i-counter]
                while t_prev ==t:
                    counter = counter+1
                    t_prev = list(df['datetimes'])[i-counter]
                
                prev_df=df[df['datetimes'] == t_prev].copy()
                highestorder = np.min(np.array(prev_df.Order))
                positions = np.array(prev_df[prev_df.Order==highestorder][flowline])
                x_prev = np.nanmean(positions) #if there are multiple grab the average
                
                if str(x_prev) == 'NaN':
                    changerate = np.NaN()
                else:
                    #calculate changes and changerate
                    dx = x-x_prev; dt = t-t_prev; dt = dt.days
                    changerate = dx/dt
            
            flowline_changes.append(changerate)
        tchange50.append(flowline_changes[0]); tchange25 = flowline_changes[1]; tchange75 = flowline_changes[2]
    df['changerate50'] = tchange50
    df['changerate25'] = tchange25
    df['changerate75'] = tchange75
    return df


# In[11]:


#define a function to calculate terminus change rate:
def calc_changerates1(df):
    import pandas as pd; import numpy as np
    df = df.dropna(subset=['tpos']) # drop any NaN terminus positions
    tchange = [] # store the terminus change rates
    
    for i in range(0, len(df.index)):
        date = list(df['datetimes'])[i]; tpos = list(df['tpos'])[i] # grab the date and terminus position
        
        #CALCULATE TERMINUS CHANGE RATE
        earliestdate = list(df['datetimes'])[0] #grab the earliest date
        #for the first date, the changerate is nan
        if date == earliestdate:
            changerate = np.NaN
        else: 
            # set current date and terminus position
            t = date; x = tpos; 
        
            #grab previous date of analysis 
            counter = 1; t_prev = list(df['datetimes'])[i-counter]
            
            # if it's the same date (previous = current), keep going back while this is true
            while t_prev == t:
                counter = counter+1; t_prev = list(df['datetimes'])[i-counter]
                
            # when previous time point is found, grab the terminus positions
            prev_df = df[df['datetimes'] == t_prev].copy(); positions = list(prev_df.tpos)
            
            #if there are multiple, grab the average or median of all of them
            #x_prev = np.nanmean(np.array(positions));
            x_prev = np.nanmedian(np.array(positions))
            
            #calculate terminus change for center (dx) in meters and time change (dt in days)
            dx = x - x_prev                  
            #calculate time change (dt) in days
            dt = t - t_prev; dt = dt.days
            #calculate change rate
            changerate = dx/dt
                   
        tchange.append(changerate);
    df['changerate'] = tchange
    return df


# In[1]:


def remove_dips(df, flow_thresh, iterations):
    import pandas as pd; import numpy as np
    for iteration in range(0, iterations):
        df = df.reset_index(drop=True); dip_indices = []; # reset indices (this will be important later)

        for index, row in df.iterrows():
            date = row['datetimes']; rate = row['changerate'] # grab date and change rate
            
            # for negative change rates:
            if rate < 0 and rate < -flow_thresh:
                # check the next entry only if it's in the range of indices
                if index+1 < len(df.index):  
                    # pick the next immediate rate & date
                    counter = 1
                    nextrate = df.loc[index+counter]['changerate']; nextdate = df.loc[index+counter]['datetimes']                    
                    # if next date is the same as the current, increment the counter
                    # to grab the next next date until the next date is different from the current
                    while nextdate == date and index+counter < len(df.index)-1:
                        counter = counter + 1; nextrate = df.loc[index+counter]['changerate']
                        nextdate = df.loc[index+counter]['datetimes']

                    # if it's a sudden jump (change rate to next > flow_thresh), then we have found a dip
                    if nextrate > abs(flow_thresh):
                        dip_indices.append(index)
                            
                # if it's a crazy large negative change, flag it even if there isn't a positive change following
                if rate < -(15*abs(flow_thresh)):
                    dip_indices.append(index) 
                    
        print("Dropping", len(dip_indices), "dips") # show number of dropped indices 
        
        # REMOVE those points and recalculate terminus change rates 
        df = df.drop(dip_indices); df = calc_changerates1(df)
    return df


# In[13]:


def remove_jumps(df, flow_thresh, iterations):
    import pandas as pd; import numpy as np
    for iteration in range(0, iterations):
        df = df.reset_index(drop=True); jump_indices = []

        for i in range(0, len(df.index)):
            date = list(df['datetimes'])[i]; rate = list(df['changerate'])[i] # grab date and change rate
            tpos = list(df['tpos'])[i]; index = list(df.index)[i] # grab the terminus position and index
            
            # if the change rate is faster than our threshold, then we have found a jump
            if rate > abs(flow_thresh):
                jump_indices.append(index)

            # grab previous date of analysis 
            counter = 1; prev_date = list(df['datetimes'])[i-counter]
            # while the previous date = current, append the counter and find the actual previous timepoint
            while prev_date == date:
                counter = counter+1; prev_date = list(df['datetimes'])[i-counter]
            
            # calculate the time between this point and the previous
            delta_date = date - prev_date; delta_date = delta_date.days

            #if the time gap is more than 2 months, and has a positive change rate
            #and the terminus position is more than 80% of the max,
            tpos_thresh = 0.8*np.max(np.array(df['tpos']))
            #remove it
            if delta_date > 60 and rate > 0:
                if tpos > tpos_thresh:
                    jump_indices.append(index)
        print("Dropping", len(jump_indices), "jumps") # show number of dropped indices 
        #drop the indices and reclaculate terminus change rates
        df = df.drop(jump_indices)
        df = calc_changerates1(df)
        
    return df


# In[5]:


#define a function to help us find the intersection of a line and a collection of points:
#determines if an input value is within a certain range/interval or a setvalue:
def within(value, setval, interval):
    if value >= setval-interval and value <= setval+interval:
        return True
    else:
        return False


# In[18]:


def distance(x1, y1, x2, y2):
    dist = (((x2-x1)**2)+((y2-y1)**2))**(1/2)
    return dist


# In[1]:


def to_datetimes(df):
    import datetime; import numpy as np
    datetimes = df.loc[:,'datetimes']; datetime_objs = []
    for date in datetimes:
        datetime_obj = datetime.datetime.strptime(str(date), '%Y-%m-%d'); datetime_obj = np.datetime64(datetime_obj)
        datetime_objs.append(datetime_obj)
    df['datetimes'] = datetime_objs
    return df


# In[10]:


def midpoint(x1, y1, x2, y2):
    midx = (x1+x2)/2; midy = (y1+y2)/2
    return midx, midy


# In[19]:


def calc_theta(manual_df):
    import pandas as pd
    import numpy as np
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
        print("Theta values:",theta)
        
                
    #CALCULATE OVERALL THETA
    theta_all = np.nanmean(thetas)
    #organize data in dataframe
    column_titles = ['Theta_avg']+BoxIDs
    theta_for_df = [theta_all]+thetas
    #write to csv
    theta_df = pd.DataFrame(list(zip(column_titles, theta_for_df)), 
                 columns=['ID', 'theta'])
    
    return theta_all



# In[4]:


def results_allglaciers(date_csv, centerline_csv, vel_csv, analysis_date, rotated_foldername, V, N1, N2):
    #import packages and functions
    import numpy as np
    import os
    import pandas as pd    
    import scipy.stats
    import datetime
    import math
    import shutil
    import subprocess
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import matplotlib.pylab as pl
    os.chdir('/home/jukes/automated-glacier-terminus')
    from automated_terminus_functions import calc_changerates1, to_datetimes, within, remove_dips, remove_jumps

    csvpaths = '/home/jukes/Documents/Sample_glaciers/'
    basepath = '/media/jukes/jukes1/LS8aws/'; massorsize = "mass"
    
    #IMAGE DATES
    datetime_df = pd.read_csv(csvpaths+date_csv, sep=',', dtype=str, header=0, names=['Scene', 'datetimes'])
    print(datetime_df.shape)
    
    #CENTERLINE INFO
    centerline_df = pd.read_csv(csvpaths+centerline_csv, sep=',', dtype=str, header=0)
    centerline_df = centerline_df.set_index('BoxID')
    
    #GLACIER VELOCITIES
    flowspeed_df= pd.read_csv(csvpaths+vel_csv, sep=',', dtype=str)
    flowspeed_df = flowspeed_df.set_index('BoxID')
    
    BoxIDs = list(pd.read_csv(csvpaths+centerline_csv, sep=',', dtype=str)['BoxID']) # List of BoxIDs
    for BOI in BoxIDs:
        print("Box"+BOI)
        metric = "Datfiles_c1/"; imagepath = basepath+"Box"+BOI+"/"+rotated_foldername
        
        order_box_df = pd.read_csv(csvpaths+'terminuspicks_Box'+BOI+'_'+analysis_date+'.csv', 
                                   sep=',', dtype=str, usecols=[1,2,3,4,0], header = 1)
#         order_box_df = order_df[order_df["BoxID"]==BOI].copy()
        order_box_df = order_box_df.drop('BoxID', axis=1)
        order_box_df = order_box_df.dropna()
        print(order_box_df.shape)

        #GRAB INFO FROM IMAGE FILES
        image_arrays = []; dats = []; trimdats = []; imgnames = []; boxids = []; scales = []
        imgfiles = os.listdir(imagepath)
        for imgfile in imgfiles:
            #grab image files and append to images list
            if imgfile.endswith(BOI+"_PS.pgm"):
                image = mpimg.imread(imagepath+imgfile); imgname = imgfile[0:-4]; scenename = imgname[2:42]
                pathtodat = imagepath+imgname+".pgm_max_gaussian/"+metric
                datfiles = os.listdir(pathtodat)
                #if there are datfiles, grab the trimmed and non-trimmed files
                if len(datfiles) > 1: 
                    #find the trimmed dat file and the original
                    for dat in datfiles:
                        if "trim" in dat:
                            datfile_trim = dat
                            #append to trimmed dats list
                            trimdats.append(datfile_trim)
                            #grab the scale and append the equivalent original dat
                            scale = dat[-7:-4]
                            datfile = "terminus_"+scale+".dat"
                            dats.append(datfile)
                            #append the image array and the image name to the list
                            image_arrays.append(image); imgnames.append(scenename); boxids.append(BOI); scales.append(scale)
        images_df = pd.DataFrame(list(zip(imgnames, boxids, image_arrays, dats, trimdats, scales)),
                      columns=['Scene','BoxID','Image_array', 'Dat_filename', "Trimmed_dat_filename", "Scale"])
        images_df.sort_values(by='Scene'); datetime_df = datetime_df.sort_values(by='Scene')

        #MERGE IMAGE INFO WITH IMAGEDATES AND MERGE WITH ORDER
        new_df = images_df.merge(datetime_df, how= 'inner', on = 'Scene')
        dated_images_df = new_df.sort_values(by='datetimes', ascending = True)
        final_images_df = dated_images_df.merge(order_box_df, how='inner', on=['Scene', 'Scale'])
        final_images_df = final_images_df.sort_values(by=['datetimes','Scene','Order'], ascending=True)

        #CALCULATE TERMINUS POSITIONS
        #LOAD IN REFERENCE POINTS to calculate terminus position with respect to
        box_midpoint_x = np.float(centerline_df.loc[BOI, 'lmid50_x']); box_midpoint_y = np.float(centerline_df.loc[BOI, 'lmid50_y'])
        boxmid_x_25 = np.float(centerline_df.loc[BOI, 'lmid25_x']); boxmid_y_25 = np.float(centerline_df.loc[BOI, 'lmid25_y'])
        boxmid_x_75 = np.float(centerline_df.loc[BOI, 'lmid75_x']); boxmid_y_75 = np.float(centerline_df.loc[BOI, 'lmid75_y'])

        #GRAB CENTERLINE POINTS
        #grab slopes and intercepts from the dataframe
        c_slope = float(centerline_df.loc[BOI]['m50']); c_intercept = float(centerline_df.loc[BOI]['b50']) 
        c25_slope = float(centerline_df.loc[BOI]['m25']); c25_intercept = float(centerline_df.loc[BOI]['b25'])
        c75_slope = float(centerline_df.loc[BOI]['m75']); c75_intercept = float(centerline_df.loc[BOI]['b75'])  

        #grab range of x-values
        xmin50 = float(box_midpoint_x); xmax50 = float(centerline_df.loc[BOI, 'rmid50_x']); ymid50 = float(box_midpoint_y)
        xmin25 = float(boxmid_x_25); xmax25 = float(centerline_df.loc[BOI, 'rmid25_x']); ymid25 = float(boxmid_y_25)
        xmin75 = float(boxmid_x_75); xmax75 = float(centerline_df.loc[BOI, 'lmid75_x']); ymid75 = float(boxmid_y_75)
        xmax = np.max([xmax50, xmax25, xmax75]); xmin = np.min([xmin50, xmin25, xmin75]); c_x = np.linspace(xmin, xmax, int(xmax-xmin)*2)

        #calculate y-values using the various centerlines
        c_y = c_slope*c_x + c_intercept; c_y_25 = c25_slope*c_x + c25_intercept; c_y_75 = c75_slope*c_x + c75_intercept

        #LISTS TO HOLD TERMINUS POSITIONS AND INTERSECTION POINTS
        terminus_positions = []; tpositions_25 = []; tpositions_75 = []
        intersections = []; X25 = []; X75 = []

        #for each scene and scale:
        for index, row in final_images_df.iterrows():
            trimdat = row['Trimmed_dat_filename']; dat = row['Dat_filename']; scene = row['Scene']    
            #CALCULATE TERMINUS POSITION
            #load in dat files and calculate intersection points
            datpath = imagepath+"R_"+scene+"_B8_Buffer"+BOI+"_PS.pgm_max_gaussian/"+metric
        #     term_trimdat = np.loadtxt(datpath+trimdat)
            term_dat=np.loadtxt(datpath+dat)
                                      
            intersect_xs = []; intersect_xs_25 = []; intersect_xs_75 = []
            intersect_ys = []; intersect_ys_25 = []; intersect_ys_75 = []

            #loop through all the x,y values for the centerline
            for j in range(0, len(c_x)):
                x = c_x[j]; y = c_y[j]; y25 = c_y_25[j]; y75 = c_y_75[j]        
                interval = 0.6
                #where are the intersections with the terminus pick?
                if len(np.shape(term_dat)) == 2: # if it's a 2D array
                    for dat_x, dat_y in term_dat:
                        #midway centerline
                        if within(dat_x, x, interval) and within (dat_y, y, interval):
                            #intersect_x = dat_x; intersect_y = dat_y; intersect_found = True
                            intersect_xs.append(dat_x); intersect_ys.append(dat_y)            
                        #1/4th centerline
                        if within(dat_x, x, interval) and within (dat_y, y25, interval):
                            intersect_xs_25.append(dat_x); intersect_ys_25.append(dat_y)              
                        #3/4th centerline
                        if within(dat_x, x, interval) and within (dat_y, y75, interval):
                            intersect_xs_75.append(dat_x); intersect_ys_75.append(dat_y)
                    
            #for 50 centerline
            #if no intersections are found with the terminus line, append Nans
            if len(intersect_xs) == 0:
                tpos50 = np.NaN; intersect_x = np.NaN; intersect_y = np.NaN
            #if at least one is found:
            else:
                #intersection with the greatest x
                #use distance formula to calculate distance between
                max_index = intersect_xs.index(np.max(intersect_xs))
                intersect_x = intersect_xs[max_index]; intersect_y = intersect_ys[max_index]
        #         term_position = distance(xmin50, ymid50, intersect_x, intersect_y)*15.0
                tpos50 = (intersect_x-xmin50)*15.0
        #         print(tpos50)

            #for 25 centerline
            if len(intersect_xs_25) == 0:
                tpos25 = np.NaN; intersect_x25 = np.NaN; intersect_y25 = np.NaN
            else:
                max_index_25 = intersect_xs_25.index(np.max(intersect_xs_25))
                intersect_x25 = intersect_xs_25[max_index_25]; intersect_y25 = intersect_ys_25[max_index_25]
                tpos25 = (intersect_x25-xmin25)*15.0
        #         tpos25 = distance(xmin25, ymid25, intersect_x25, intersect_y25)*15.0

            #for 75 centerline
            if len(intersect_xs_75) == 0:
                tpos75 = np.NaN; intersect_x75 = np.NaN; intersect_y75 = np.NaN
            else:
                max_index_75 = intersect_xs_75.index(np.max(intersect_xs_75))
                intersect_x75 = intersect_xs_75[max_index_75]; intersect_y75 = intersect_ys_75[max_index_75]
                tpos75 = (intersect_x75-xmin75)*15.0
        #         tpos75 = distance(xmin75, ymid75, intersect_x75, intersect_y75)*15.0

            #append to lists
            terminus_positions.append(tpos50); tpositions_25.append(tpos25); tpositions_75.append(tpos75)
            intersections.append([intersect_x, intersect_y]); X25.append([intersect_x25, intersect_y25]); X75.append([intersect_x75, intersect_y75])

        # ADD TERMINUS POSITION AND INTERSECTIONS
        final_images_df['tpos50'] = terminus_positions; final_images_df['tpos25'] = tpositions_25; final_images_df['tpos75'] = tpositions_75
        final_images_df['X50'] = intersections ;final_images_df['X25'] = X25; final_images_df['X75'] = X75

        #SPLIT INTO 3 DATAFRAMES FOR 3 FLOWLINES:
        final_images_50 = final_images_df[['Scene', 'BoxID', 'Scale', 'datetimes', 'Metric', 'Order', 
                                          'tpos50', 'X50',]].copy().reset_index(drop=True)
        final_images_50 = final_images_50.rename(columns={"tpos50": "tpos", "X50": "X"})
        final_images_25 = final_images_df[['Scene', 'BoxID', 'Scale', 'datetimes', 'Metric', 'Order', 
                                          'tpos25', 'X25']].copy().reset_index(drop=True)
        final_images_25 = final_images_25.rename(columns={"tpos25": "tpos", "X25": "X"})
        final_images_75 = final_images_df[['Scene', 'BoxID', 'Scale', 'datetimes', 'Metric', 'Order', 
                                          'tpos75', 'X75']].copy().reset_index(drop=True)
        final_images_75 = final_images_75.rename(columns={"tpos75": "tpos", "X75": "X"})
        dfs = [final_images_50, final_images_25, final_images_75]

        #CALCULATE TERMINUS CHANGE RATES
        dfs_new = []
        for df in dfs: 
            to_datetimes(df); dfs_new.append(calc_changerates1(df))

        #FILTER USING 5*MAXIMUM FLOW SPEEDS
        max_flow = float(flowspeed_df['Max_speed'][BOI])
        if max_flow < 1.0:
            flow_thresh = V
        else:
            flow_thresh = V*max_flow
        #remove dips
#         N1 = 3; 
        nodips = []
        for df in dfs_new:
            nodips.append(remove_dips(df, flow_thresh, N1))
        #remove jumps
#         N2 = 2; 
        nojumps = []
        for df in nodips:
            nojumps.append(remove_jumps(df, flow_thresh, N2))
        
        stop = 0
        #stop the process if there are no points
        for df in nojumps:
            if len(df) == 0:
                stop = 1
                # print('No points remaining. Processed stopped for Box'+BOI)
                
        if stop == 0:
            #GRAB HIGHEST ORDER PICK AFTER FILTERING
            highestorder_dfs = []
            for df in nojumps:
                    #grab unique dates
                    unique_dates = set(list(df['datetimes']))
                    # print(len(unique_dates))
                    #grab highest orders:
                    order_list = []
                    for date in unique_dates:
                        date_df = df[df['datetimes'] == date].copy()
                        highestorder = np.min(np.array(date_df['Order']))
                        order_list.append(highestorder)
                    highestorder_df = pd.DataFrame(list(zip(unique_dates, order_list)), columns=['datetimes', 'Order']).sort_values(by='datetimes', ascending=True)
                    highestorder_dfs.append(highestorder_df)

            onepick_dfs = []
            for i in range(0, len(highestorder_dfs)):
                onepick_df = nojumps[i].merge(highestorder_dfs[i], how='inner', on=['datetimes', 'Order'])
                onepick_dfs.append(onepick_df)
                # print(onepick_df.shape[0])

            #PLOT AND SAVE
            colors = pl.cm.viridis(np.linspace(0,0.5,3)) # generate 3 centerline 
            fig, ax1 = plt.subplots(figsize=(12,4))
            
            for j in range(0, len(onepick_dfs)):
                df = onepick_dfs[j];    print(len(df))
                ax1.plot(df['datetimes'], df['tpos'], color=colors[j], marker='o', markersize=6, linewidth=0, alpha=0.7)
            #general plot parameters
            ax1.set_ylabel('Terminus position (m)', color='k', fontsize=12)
            ax1.set_title("Box"+BOI, fontsize=16); ax1.set_xlabel('Date', fontsize=12)
            ax1.tick_params(axis='both', which='major', labelsize=12)
            #save figure
            plt.legend(['1/2', '1/4', '3/4'])
            plt.savefig(csvpaths+"/Figures/Termposition_LS8_m_Box"+BOI+"_"+analysis_date+".png", dpi=200)
            plt.show()

            flowlines = ['flowline50', 'flowline25', 'flowline75']
            for k in range(0, len(onepick_dfs)):
                df = onepick_dfs[k];
                df.to_csv(path_or_buf = csvpaths+'Tpos_Box'+BOI+'_'+flowlines[k]+'_filtered.csv', sep=',')



# In[ ]:

def resize_imgs(path, iarray):
    import numpy as np
    import os
    from PIL import Image
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    
    dimensions_x = []; dimensions_y = []
    images = os.listdir(path)
    for image in images:
        if image.endswith('.TIF'):
            img = mpimg.imread(path+image)
            dimensions_x.append(img.shape[1]); dimensions_y.append(img.shape[0])

    #find minimum dimensions
    min_y = np.min(dimensions_y); min_x = np.min(dimensions_x)
    index_y = dimensions_y.index(min_y); index_x = dimensions_x.index(min_x)
    
    if (index_x != index_y): # if the minimum dimensions don't match:
        if (index_x != 0) or (index_y != 0): # minimum is found:
             # select the dimension that does have a correct minimum
            if index_x != 0:
                index_y = index_x
            elif index_y != 0:
                index_x = index_y
        else:
            print('Something is funky with the image dimensions:')
            print('X:',list(dimensions_x))
            print('Y:', list(dimensions_y))
            print(index_x)
            print(index_y)

    if (index_x == index_y): # now if they match
        # crop the iarray:
        if iarray.shape[1] > min_x or iarray.shape[0] > min_y:
            diffx_half = (iarray.shape[1] - min_x)/2; diffy_half = (iarray.shape[0] - min_y)/2

            # if the difference is a half pixel, make sure to remove the full value from the first side only
            if int(diffx_half) != diffx_half:
                #remember for image slicing y is the first dimension, x is the second
                iarray_cropx = iarray[:,int(diffx_half):-int(diffx_half)-1,:] 
            else: #otherwise remove it from both sides:
                iarray_cropx = iarray[:, int(diffx_half):-int(diffx_half),:]

            #same for y
            if int(diffy_half) != diffy_half:   
                iarray_cropy = iarray_cropx[int(diffy_half):-int(diffy_half)-1,:,:]
            else:
                iarray_cropy = iarray_cropx[int(diffy_half):-int(diffy_half),:,:]

            print(min_y, min_x, iarray_cropy.shape)
            return iarray_cropy
        else:
            print(min_y, min_x, iarray.shape)
            return iarray
        
        # crop the images
        for image in images:
#         # crop each image if the dimensions are larger than the minimum)
            if image.endswith('.TIF'):
                img = mpimg.imread(path+image)
                if img.shape[1] > min_x or img.shape[0] > min_y:
                    print(image, 'cropped')
                    #calculate difference, and divide by 2 to get amount of rows to remove by
                    diffx_half = (img.shape[1] - min_x)/2; diffy_half = (img.shape[0] - min_y)/2

                    #if the difference is a half pixel, make sure to remove the full value from the first side only
                    if int(diffx_half) != diffx_half:
                        #remember for image slicing y is the first dimension, x is the second
                        img_cropx = img[:, int(diffx_half):-int(diffx_half)-1]
                    else: #otherwise remove it from both sides:
                        img_cropx = img[:, int(diffx_half):-int(diffx_half)]

                    #same for y
                    if int(diffy_half) != diffy_half:   
                        img_cropy = img_cropx[int(diffy_half):-int(diffy_half)-1, :]
                    else:
                        img_cropy = img_cropx[int(diffy_half):-int(diffy_half), :]
                    
                    #save over original images
                    resized = np.ascontiguousarray(img_cropy)
                    plt.imsave(path+image[:-4]+'.TIF', resized, cmap='gray')

# In[ ]:
def terminuspick_1glacier(BoxID, inputs, CPU):
    import subprocess
    order = inputs[0]
    size_thresh = inputs[1]
    mod_thresh = inputs[2]
    arg_thresh = inputs[3]
    tpick = 'taskset '+CPU+' /home/akhalil/src/xsmurf-2.7/main/xsmurf -nodisplay /home/jukes/Documents/Scripts/terminus_pick'+str(order)+'_1glacier.tcl '+str(size_thresh)+' '+str(mod_thresh)+' '+str(arg_thresh)+' '+str(BoxID)
    out = subprocess.Popen(tpick, shell=True)
    PID = out.pid
    print(PID+1)
    
# In[ ]:
def terminuspick_1glacier_opt(BoxID, inputs, CPU):
    import subprocess
    order = inputs[0]
    size_thresh = inputs[1]
    mod_thresh = inputs[2]
    arg_thresh = inputs[3]
    dataset = inputs[4]
    tpick = 'taskset '+CPU+' /home/akhalil/src/xsmurf-2.7/main/xsmurf -nodisplay /home/jukes/Documents/Scripts/terminus_pick'+str(order)+'_1glacier.tcl '+str(size_thresh)+' '+str(mod_thresh)+' '+str(arg_thresh)+' '+str(BoxID)+' '+str(dataset)
    out = subprocess.Popen(tpick, shell=True)
    PID = out.pid
    print(PID+1)


# In[20]:


def objective_func(size_thresh, mod_thresh, arg_thresh, order, dataset, V, N1, N2, date):
    import subprocess
    import os
    import pandas as pd
    from multiprocessing import Barrier
    os.chdir('/home/jukes/automated-glacier-terminus')
    from automated_terminus_functions import results_allglaciers, terminuspick_1glacier_opt, parallel
    
    num_CPUs = 8; BoxIDs = ['001', '002', '120', '174', '259']
    num_glaciers = len(BoxIDs); 

    parallel_mp(num_CPUs, num_glaciers, terminuspick_1glacier_opt, [order, size_thresh, mod_thresh, arg_thresh, dataset], BoxIDs, True)
    
    # from the lines, get the results using the new result_all glaciers function:
    results_allglaciers('Images_downloaded_sample5.csv','imgdates_sample10.csv', 'Boxes_coords_pathrows_sample5.csv','Glacier_vel_measures_sample10.csv', date, V,N1,N2)
    
    #calculate value of theta
    manual_df = pd.read_csv('/media/jukes/jukes1/Manual/manual_tpos_c1.csv', dtype=str,sep=',')
    return calc_theta(manual_df)

# In[ ]:
def scr_gaussian_1image(image, BoxID, CPU):
    import subprocess
    scr = 'taskset '+CPU+' /home/akhalil/src/xsmurf-2.7/main/xsmurf -nodisplay /home/jukes/Documents/Scripts/scr_gaussian_image.tcl '+BoxID+' '+image
    out = subprocess.Popen(scr, shell=True)
    PID = out.pid
    return(PID+1)
    
# In[ ]:
def parallel_mp(num_CPUs, num_items, target, targetargs, listofitems, barrier):
    import math
    from multiprocessing import Process, Barrier
    import subprocess
    import psutil
    CPUs = ['0x1', '0x2', '0x4', '0x8', '0x10', '0x20', '0x40', '0x80']
    num_batches = math.ceil(num_items/num_CPUs)
    num_lastbatch = num_CPUs - (num_batches*num_CPUs - num_items)
    
    counter = 0
    for i in range(1, num_batches+1):
        if i < num_batches:
                print("Batch", i)
                #begin multiproccess
                jobs = []; PIDs = []
                for j in range(1, num_CPUs+1):
                    item = listofitems[counter] # grab the item
                    p = Process(target=target, args=(item, targetargs, CPUs[j-1]))
                    p.start() # start the job for the image
                    jobs.append(p)
                    PIDs.append(p.pid)
                    print("CPU", j, ':', item); 
                    counter = counter+1
                
                for l in range(0, len(jobs)):
                    if psutil.pid_exists(PIDs[l]):
                        job = jobs[l]; job.join() # wait on all the jobs to finish
        else:
                print("Batch", i)
                jobs_lb = []; PIDs_lb = []
                for k in range(1, num_lastbatch+1):
                    item = listofitems[counter] # grab the item
                    p = Process(target=target, args=(item, targetargs, CPUs[k-1]))
                    p.start() # start the job for the image
                    jobs_lb.append(p)
                    PIDs_lb.append(p.pid)
                    print("CPU", k,':', item); 
                    counter = counter+1
                for q in range(0, len(jobs_lb)):
                    if psutil.pid_exists(PIDs_lb[q]):
                        job = jobs_lb[q]; job.join() # wait on all the jobs to finish
#                 for job in jobs_lb:
#                     job.join() # wait on jobs to finish
    if barrier == True:
        b = Barrier(num_lastbatch-1)
        b.wait()
        
# In[ ]:

def parallel(num_CPUs, num_items, target, targetargs, listofitems):
    import math
    import subprocess
    os.chdir('/home/jukes/automated-glacier-terminus')
    from automated_terminus_functions import target
    
    num_batches = math.ceil(num_items/num_CPUs)
    num_lastbatch = num_CPUs - (num_batches*num_CPUs - num_items)
    
    counter = 0
    for i in range(1, num_batches+1):
        if i < num_batches:
                print("Batch", i)
                #begin multiproccess
                jobs = []
                for j in range(1, num_CPUs+1):
                    item = listofitems[counter] # grab the item
                    PID = scr_gaussian_1image(item, BoxID)
                    jobs.append(PID)
                    print("CPU", j, ':', item); 
                    counter = counter+1
                
                for job in jobs:
                    job.join() # wait on all the jobs to finish
        else:
                print("Batch", i)
                jobs_lb = []
                for k in range(1, num_lastbatch+1):
                    item = listofitems[counter] # grab the item
                    p = Process(target=target, args=(item, targetargs))
                    p.start() # start the job for the image
                    jobs_lb.append(p)
                    print("CPU", k,':', item); 
                    counter = counter+1
                
                for job in jobs_lb:
                    job.join() # wait on jobs to finish
    if barrier == True:
        b = Barrier(num_lastbatch-1)
        b.wait()
        



