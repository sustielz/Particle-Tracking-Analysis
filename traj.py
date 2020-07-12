import cv2, json
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import trackpy as tp


#### This file includes various methods for data analysis
def load(filename, path='', refined=False):  ## Load data from json file
    filename = filename.split('.')[0]  ## Precaution, in case filename includes suffix
    if path is not '' and path[-1] is not '/':
        path = path + '/'
    if refined:
        with open(path+'refined/'+filename+'.json', 'r') as f:
            preds =  json.load(f)
    else:
        with open(path+'MLpreds/'+filename+'.json', 'r') as f:
            preds =  json.load(f)
    return pd.DataFrame.from_dict(preds)


################ Data Analysis Functions ########################
######## Methods for trajectories ########

###### Getting Trajectories ######
def link(dat, memory=5, maxdis=50): ## Use trackpy.link_df to find trajectories     
    dat = dat.rename(columns={"x_p": "x", "y_p": "y", "framenum": "frame"})
    dat = tp.link_df(dat, maxdis, memory=memory)
    return dat.rename(columns={"x": "x_p", "y": "y_p", "frame": "framenum"})
 
###### Swapping trajectories ######
#### Swap trajectories i and j
def swap_traj(dat, i, j):
#     dat.temp = dat.particle
#     dat.loc[dat['particle']==i, 'temp'] = j
#     dat.loc[dat['particle']==j, 'temp'] = i
#     dat.particle = dat.temp 
#     del dat['temp']
#     return dat.astype({'col1': 'int32'}).dtypes
    dat['particle'] = dat['particle'].map( {i:j, j:i} )
#     return dat

#### Re-order particles into given index (i.e. particle i -> index[i])
def reorder_traj(dat, index):
    dat['particle'] = dat['particle'].map( {index[i]:i for i in range(len(index))} )
#     return dat

#### Re-index the particles based on their (average) value of some property 
def sort_traj(dat, prop='x_p'):   #### By default, sort particles from left to right
    vals = [np.mean(x) for x in get_prop(dat, prop)]
#     return reorder_traj(dat, np.argsort(vals))
#     print(dat[{'x_p', 'particle'}])
    reorder_traj(dat, np.argsort(vals))
#     print(dat[{'x_p', 'particle'}])
#     print(vals)
#     print(np.argsort(vals))
###### Removing Trajectories ######
def remove_trajs(dat, index):
    index.sort(reverse=True)
    for i in index:
        dat.loc[dat['particle']==i, 'particle'] = -1
        dat.loc[dat['particle']>i, 'particle'] -= 1
#     return dat
   
def filter_trajlen(dat, minlength=200):
#     return remove_trajs(dat, [i for i in range(max(dat.particle)+1) if (np.size(dat.loc[dat['particle']==i, 'x_p']) < minlength) ])
    remove_trajs(dat, [i for i in range(max(dat.particle)+1) if (np.size(dat.loc[dat['particle']==i, 'x_p']) < minlength) ])
               
    
###### Getting Properties ######
#### In: dataframe, property, i --> Out: property of particle i.    If i is None (default), return list of all particles
#### Recognizes some custom 'keywords' (i.e. 'r' returns radius, appending '0' subtracts mean)
def get_prop(dat, prop, i=None):
    if i is None:    #### if i is None (default), return list: Out[i] = property of particle i
        return [get_prop(dat, prop, i=i) for i in range( max(dat.particle)+1 )]

    if prop[0] is 'r':    ## Read 'radius'
        x = get_prop(dat, 'x_p' + prop[1:], i=i)
        y = get_prop(dat, 'y_p' + prop[1:], i=i)
        return np.sqrt(x**2 + y**2)
    
    elif prop[-1] is '0': ## Read 'without mean'
        x = get_prop(dat, prop[:-1], i=i)
        return x - np.mean(x) 
    
    else:                 ## Base case: get the property directly from dataframe
        return dat.loc[dat['particle']==i, prop].to_numpy()  

    
    
# #### Input: list of properties, list of dataframes, number of dataframes (length of list), number trajs to keep
# #### Output: 3D list OUT[i][j][k] is property i of dataframe j for particle k
# def runs2props(runs, props, sortprop='x_p'):
#     runs = [sort_traj(run, sortprop) for run in runs]
#     return [ [get_prop(runs[i], prop) for i in range(len(runs))] for prop in props]


# def scatter_traj(dat):
#     X = get_prop(dat, 'x_p')
#     Y = get_prop(dat, 'y_p')
#     for i in range(len(X)):
#         plt.scatter(X[i], Y[i], label='traj. {} ({} points)'.format(i, np.size(X[i])))
#     plt.legend()

# A more specific function to make loading easier. Recommended for user to define their own 'quickload' (etc) function.
def quickload(filename, scatter=False, path=''):
    dat = load(filename, path=path)
    link(dat)
    filter_trajlen(dat)
    if(scatter):
        scatter_traj(dat)
    return dat

