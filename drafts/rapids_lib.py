# measurement/logging
import time

# data arrays/frames
import numpy as np
import pandas as pd
import cudf

# algos, datasets
import xgboost; from xgboost import plot_tree
import cuml
import sklearn; from sklearn import datasets; from sklearn.metrics import confusion_matrix, accuracy_score

# gpu libs
import cupy
import numba; from numba import cuda

'''
-------------------------------------------------------------------------
>  DATA GEN
    - done with cupy.random.normal, cuml.make_blobs in RAPIDS v0.10
------------------------------------------------------------------------- 
'''
def convert_to_cuDFs ( data, labels ):
    '''  build cuda DataFrames for data and labels from cupy arrays '''
    labels_cDF = cudf.DataFrame([('labels', labels)])
    data_cDF = cudf.DataFrame([('x', cupy.asfortranarray(data[:,0])), 
                               ('y', cupy.asfortranarray(data[:,1])), 
                               ('z', cupy.asfortranarray(data[:,2]))])
    return data_cDF, labels_cDF

def generate_galaxy_dataset ( nSamples = 1000000, 
                             nGalaxies = 5, nPlanets = 5, nLabels = 5, 
                             shuffleFlag = True, 
                             rSeed = None ):
    ''' build random galaxy dataset on the GPU
    
        galaxy = set of 3D random sized blobs, randomly offset from a common center

        --- inputs --- 
            nSamples : total number of samples in the entire dataset [in all galaxies]
            nGalaxies : number of galaxies [ see definition above ]
            nClusters : number of blobs within a galaxy
            shuffleFlag : whether the blob samples should be randomly permuted
            rSeed : random seed for repeatability

        --- returns --- 
            X : cuda DataFrame 3D points
            y : cuda DataFrame label
            t : elapsed time  
    '''
    if rSeed is not None:
        cupy.random.seed(rSeed)
        
    print('generating shuffled samples n = {}'.format(nSamples))
    startTime = time.time()
    
    nClusterPoints = nSamples // ( nPlanets * nGalaxies )
    data = labels = None
    
    # generate galaxies
    for iGalaxy in range ( nGalaxies ):
        galaxyCenter = cupy.random.randint( 0, nGalaxies**2 + nGalaxies ) # randomly choose a galaxy center
        # generate blob/planet within a galaxy
        for iPlanet in range( nPlanets ):
            
            # generate blob using 3D unifrom normal distribution with ranomized standard deviation
            clusterData =  cupy.random.normal( loc = galaxyCenter, 
                                               scale = cupy.random.randn()/4.5, 
                                               size = (nClusterPoints, 3) ) 
            
            # 3D shift each blob/planet to a unique location in the 'galaxy'
            clusterData += cupy.random.randn(3)
            
            # assign class / label
            clusterLabels = cupy.ones((nClusterPoints), dtype='int32') * iPlanet
            
            if data is None and labels is None: 
                data = clusterData; labels = clusterLabels
            else:
                data = cupy.concatenate((data, clusterData))
                labels = cupy.concatenate((labels, clusterLabels))
                
    # optionally shuffle [ default == shuffle ]
    if shuffleFlag:
        shuffledInds = cupy.random.permutation (nSamples)
        data_cDF, labels_cDF = convert_to_cuDFs ( data[shuffledInds], labels[shuffledInds])
    else:
        data_cDF, labels_cDF = convert_to_cuDFs ( data, labels)
        
    return data_cDF, labels_cDF, time.time() - startTime


''' -------------------------------------------------------------------------
>  VISUALIZATION
------------------------------------------------------------------------- 
'''
import ipyvolume as ipv
import importlib; interactivePlottingLibSpec = importlib.util.find_spec("ipyvolume")
import matplotlib.pyplot as plt; from mpl_toolkits.mplot3d import Axes3D
import warnings; warnings.filterwarnings('ignore');

def visualize_data ( data, colors = 'purple', maxSamplesToPlot = 50000, colorMapName = 'tab10' ):
    
    maxSamplesToPlot = min( ( data.shape[0], maxSamplesToPlot ) )
    stride = data.shape[0]//maxSamplesToPlot    
    
    # trim data and send to host/CPU [ via Pandas ] for plotting in the browser     
    if type(data) == pd.DataFrame: data = data[::stride].as_matrix()
    else: data = data[::stride].to_pandas().as_matrix()
        
    # convert indecies to colors and send to host/CPU [ whenever the color argument is not a string ]
    if not isinstance( colors, str):           
        if type(colors) != np.ndarray: colorStack = np.squeeze(colors[::stride].to_pandas().as_matrix())
        else: colorStack = np.squeeze(colors[::stride])            
        cMap = np.matrix( plt.get_cmap( colorMapName ).colors)
        colors = cMap[colorStack.astype(int)].astype( 'float64')
    
    if interactivePlottingLibSpec is not None: # and mode.strip().lower() != '2d': 
        ipv_plot_data(data.astype( 'float64'), colorStack = colors)
    else: 
        mpl_plot_data(data.astype( 'float64'), colorStack = colors)
    print('plotting ', maxSamplesToPlot, 'out of', data.shape[0], ' [ stride = ', stride, ']')
    
def mpl_plot_data( data, colorStack = 'purple', ax3D = False, markerScale=.1):
    if not ax3D: ax3D = plt.figure(figsize=(15,15)).gca(projection='3d')
    ax3D.scatter(data[:,0], data[:,1], data[:,2], s = 20*markerScale, c=colorStack, depthshade=True)
    ax3D.view_init(elev=15, azim=15)
    return ax3D
    
def ipv_plot_data( data, colorStack = 'purple', holdOnFlag = False, markerSize=.25):
    if not holdOnFlag: ipv.figure(width=600,height=600) # create a new figure by default, otherwise append to existing    
    ipv.scatter( data[:,0], data[:,1], data[:,2], size = markerSize, marker = 'sphere', color = colorStack)
    if not holdOnFlag: ipv.show()    
        
'''
-------------------------------------------------------------------------
>  MEASUREMENT / LOGGING
------------------------------------------------------------------------- 
'''

import re
def update_log( log, logEntries ):
    for iEntry in logEntries:
        iLabel = iEntry[0]; iValue = iEntry[1]
        if iLabel in log.keys():
            nMatchingKeys = sum([1 for key, value in log.items() if re.search('^'+iLabel, key)])
            iLabel = iLabel + '_' + str(nMatchingKeys)
        log[iLabel] = iValue
        print(' + adding log entry [ {0:25s}:{1:10.5f} s ]'.format(iLabel, iValue))
    
    return log

def query_log ( log, queryKey, verbose = False ):
    values = [value for key, value in log.items() if re.search('^'+queryKey, key) ]
    if verbose:
        print(queryKey, '\n', pd.Series(values).describe(), '\n',)    
        plt.plot(values, 'x-')
    return values

'''
-------------------------------------------------------------------------
>  CHOICE CONFIRMATION
------------------------------------------------------------------------- 
'''
def accept_choices( nSamples, dataFormat, graphicsMode ):    
    assert(type(nSamples) == int )
    assert(type(dataFormat) == str and dataFormat.strip().lower() in ['csv','parquet'] )
    assert(type(graphicsMode) == str and graphicsMode.strip().lower() in ['2d', '3d'] )
    
    print( 'Below are your current choices for the key parameters in the notebook, return to the cell above to revise them. \n' )    
    print( 'Number of Total Samples:'.rjust(40, ' '), ' {}'.format( nSamples ))
    print( 'Data Format [ CSV or parquet ]:'.rjust(40, ' '), ' {}'.format( dataFormat ))    
    print( 'GraphicsMode [ 2D or 3D ]:'.rjust(40, ' '), ' {}'.format( graphicsMode ))    
    '''
    yN = input( "\nAre all these values correct? [type 'y' for yes, or 'n' for no ]: ")
    if str(yN).lower().strip() != ('y' or 'yes') : 
        raise ValueError ( '! hmm, unacceptable choices ahead...proceed with caution ' )
    else: print('\n Choices confirmed, lets move forward :]')
    '''

'''
-------------------------------------------------------------------------
>  VIZ PERF
------------------------------------------------------------------------- 
''' 
def viz_perf ( expTimesDF, yLim ):
    
    categories = list(expTimesDF.columns[1:])

    values = expTimesDF.values.flatten().tolist()
    values += values[:1] # repeat the first value to close the circular graph

    angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
    angles += angles[:1]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.set_facecolor('white')
    plt.xticks(angles[:-1], categories, color='grey', size=12)
    #plt.yticks(np.arange(1, 6), ['1', '2', '3', '4', '5'], color='grey', size=12)
    plt.ylim(0, yLim)
    ax.set_rlabel_position(30)

    # part 1
    val_c1 = expTimesDF.loc[0].drop('experiment_id').values.flatten().tolist()
    val_c1 += val_c1[:1]
    ax.plot(angles, val_c1, linewidth=1, linestyle='solid', label='CPU')
    ax.fill(angles, val_c1, 'skyblue', alpha=0.4)

    val_c2 = expTimesDF.loc[1].drop('experiment_id').values.flatten().tolist()
    val_c2 += val_c2[:1]
    ax.plot(angles, val_c2, linewidth=1, linestyle='solid', label='GPU')
    ax.fill(angles, val_c2, 'purple', alpha=0.1)


    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.show()    
    