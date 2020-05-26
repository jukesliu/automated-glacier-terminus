#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/python

import sys

inputs=sys.argv[1:]
print(inputs)

if len(inputs) != 9:
    print("Incorrect number of input arguments.")
else:
    # read in inputs
    download_csv = inputs[0]
    date_csv = inputs[1]
    centerline_csv = inputs[2]
    vel_csv = inputs[3]
    analysis_date = inputs[4]
    V = int(inputs[5]); N1 = int(inputs[6]); N2 = int(inputs[7])
    BoxID = inputs[8]
    
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
    
    for BOI in [BoxID]:
        print("Box"+BOI+ "results analysis")
        metric = "Datfiles_c1/"; imagepath = basepath+"Box"+BOI+"/rotated_c1/"
        
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
        #         for dat_x, dat_y in term_trimdat:
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
            fig, ax1 = plt.subplots(figsize=(12,4))
            markers = ['mo', 'ro', 'bo']
            for j in range(0, len(onepick_dfs)):
                df = onepick_dfs[j];    print(len(df))
                ax1.plot(df['datetimes'], df['tpos'], markers[j], markersize=5, alpha=0.7)
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

