#!/usr/bin/env python
# coding: utf-8

# # Functions written for automated-glacier-terminus detection
# Jukes Liu. _Github: julialiu18_
# 
#  - calc_changerates3
#  - calc_changerates1
#  - remove_dips
#  - remove_jumps
#  - within
#  - distance
#  - to_datetimes
#  - midpoint

# In[1]:


#define a function to calculate terminus change rate:
def calc_changerates3(df):
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


# In[2]:


#define a function to calculate terminus change rate:
def calc_changerates1(df):
    df = df.dropna(subset=['tpos'])
    original_len = df.shape[0]; tchange= []
    
    for i in range(0, len(df.index)):
        date = list(df['datetimes'])[i]; tpos = list(df['tpos'])[i]
        
        #CALCULATE TERMINUS CHANGE RATE
        #grab the earliest date
        earliestdate = list(df['datetimes'])[0]
        #for the first date, the changerate is nan
        if date == earliestdate:
            changerate = np.NaN
        #for all other subsequent entries:
        else:
            #grab current date and terminus positions
            t = date; x = tpos; 
        
            #grab previous date of analysis 
            counter = 1; t_prev = list(df['datetimes'])[i-counter]
            #while the previous date = current date, append the counter and find the date before that!
            while t_prev == t:
                counter = counter+1; t_prev = list(df['datetimes'])[i-counter]

            #grab all terminus positions from previous date of analysis:
            prev_df = df[df['datetimes'] == t_prev].copy()
            positions = list(prev_df.tpos)
            #if there are multiple, grab the average of all of them
            x_prev = np.nanmean(np.array(positions));
            
            #calculate terminus change for center (dx) in meters and time change (dt in days)
            dx = x - x_prev                  
            #calculate time change (dt) in days
            dt = t - t_prev; dt = dt.days
            #calculate change rate
            changerate = dx/dt
                   
        tchange.append(changerate);
    df['changerate'] = tchange
    return df


# In[3]:


def remove_dips(df, flow_thresh, iterations):
    for iteration in range(0, iterations):
        #reset index
        df = df.reset_index(drop=True)
        dip_indices = [];

        # for index, row in onepick_df.iterrows():
        for index, row in df.iterrows():
            date = row['datetimes']
            rate = row['changerate']
            #for negative change rates:
            if rate < 0 and rate < -flow_thresh:
                #check the next entry only if it's in the range of indices
                if index+1 < len(df.index):  
                    counter = 1
                    #pick the next immediate rate/date
                    nextrate= df.loc[index+counter]['changerate']; nextdate = df.loc[index+counter]['datetimes']                    
                    #while the next date is the same as the current, increment the counter
                    #to grab the next next date until the next date is different from the current
                    while nextdate == date and index+counter < len(df.index)-1:
                        counter = counter + 1; nextrate = df.loc[index+counter]['changerate']
                        nextdate = df.loc[index+counter]['datetimes']

                    #if it's a sudden jump, then we have found a dip. Remove it
                    if nextrate > abs(flow_thresh):
                        dip_indices.append(index)
                            
                #if it's a crazy large negative change, 
                #remove it even if there isn't a positive change following
                if rate < -(15*abs(flow_thresh)):
                    dip_indices.append(index)            
        print(dip_indices)
        #REMOVE THOSE TERMINUS POSITIONS
        df = df.drop(dip_indices)
        #recalculate terminus changerates
        df = calc_changerates1(df)
    return df


# In[4]:


def remove_jumps(df, flow_thresh, iterations):
    for iteration in range(0, iterations):
        #reset index for final_images_df
        df = df.reset_index(drop=True); jump_indices = []

        for i in range(0, len(df.index)):
            date = list(df['datetimes'])[i]; rate = list(df['changerate'])[i]
            tpos = list(df['tpos'])[i]; index = list(df.index)[i]

            if rate > abs(flow_thresh):
                #remove it:
                jump_indices.append(index)

            #remove drops if they are due to first value for the season
            #grab previous date of analysis 
            counter = 1; prev_date = list(df['datetimes'])[i-counter]
            #while the previous date = current date, append the counter and find the previous previous date
            while prev_date == date:
                counter = counter+1; prev_date = list(df['datetimes'])[i-counter]
            delta_date = date - prev_date; delta_date = delta_date.days

            #if the time gap is more than 2 months, and has a positive change rate
            #and the terminus position is more than 80% of the max,
            tpos_thresh = 0.8*np.max(np.array(df['tpos']))
            #remove it
            if delta_date > 60 and rate > 0:
                if tpos > tpos_thresh:
                    jump_indices.append(index)
        print(jump_indices)
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


# In[13]:


def distance(x1, y1, x2, y2):
    dist = (((x2-x1)**2)+((y2-y1)**2))**(1/2)
    return dist


# In[7]:


def to_datetimes(df):
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


# In[ ]:




