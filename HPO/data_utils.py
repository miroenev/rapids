import numpy as np; import numpy.matlib
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipyvolume as ipv

import time
import copy
import gzip 

import cupy
import cudf
import cuml

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets; from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

import dask
from dask import delayed
import xgboost

import pickle

import os
import sys
from enum import Enum

from urllib.request import urlretrieve

rapidsPrimary1 = [ 116/255., 0/255., 255/255., 1]
nvidiaGreen = [ 100/255., 225/255., 0., 1]
rapidsPrimary2 = [ 255/255., 181/255., 0/255., 1]
rapidsPrimary3 = [ 210/255., 22/255., 210/255., 1]
rapidsSecondary4 = [ 0., 0., 0., 1]
rapidsSecondary5 = [ 102/255., 102/255., 102/255., 1]

rapidsColors = { 0: rapidsPrimary1,
                 1: nvidiaGreen,
                 2: rapidsPrimary2,
                 3: rapidsPrimary3,
                 4: rapidsSecondary4,
                 5: rapidsSecondary5 }



class Dataset():    
    def __init__(self, datasetName = 'synthetic', nSamples = None):
        self.datasetName = datasetName
        
        if self.datasetName == 'synthetic':
            
            if nSamples == None: nSamples = 1000000
            data, labels, elapsedTime  = generate_dataset( coilType = 'helix', nSamples = nSamples)
            self.trainObjective = ['binary:hinge', None]

        elif self.datasetName == 'fashion-mnist':

            data, labels, elapsedTime = load_fashion_mnist () 
            self.trainObjective = ['multi:softmax', 10]

        elif self.datasetName == 'airline':
            
            if nSamples == None: nSamples = 5000000
            data, labels, elapsedTime = load_airline_dataset ( 'data/', np.min( ( nSamples, 115000000 )))
            self.trainObjective = ['binary:hinge', None]
        
        # split train and test data
        self.trainData, self.trainLabels, self.testData, self.testLabels = self.split_train_test ( data, labels )
        
        # apply standard scaling
        trainMeans, trainSTDevs, _ = scale_dataframe_inplace ( self.trainData )
        _,_,_ = scale_dataframe_inplace ( self.testData, trainMeans, trainSTDevs )
        
    def split_train_test(self, data, labels, trainSize = .75, trainTestOverlap = .025 ):
        if self.datasetName == 'synthetic':            
            trainData, trainLabels, testData, testLabels, _ = split_train_test_nfolds ( data, labels, trainTestOverlap = trainTestOverlap )
        else:
            trainData, testData, trainLabels, testLabels = cuml.train_test_split( data, labels, shuffle = False, train_size= trainSize )        
        return trainData, trainLabels, testData, testLabels

''' -------------------------------------------------------------------------
>  DATA LOADING
------------------------------------------------------------------------- '''
def data_progress_hook ( blockNumber, readSize, totalFileSize ):
    if ( blockNumber % 1000 ) == 0:        
        print(f' > percent complete: { 100 * ( blockNumber * readSize ) / totalFileSize:.2f}\r', end='')
    return

def download_dataset ( url, localDestination ):
    if not os.path.isfile( localDestination ):
        print(f'no local dataset copy at {localDestination}')
        print(f' > downloading dataset from: {url}')
        urlretrieve( url = url, filename = localDestination, reporthook = data_progress_hook )

def load_higgs_dataset (dataPath='./data/higgs', nSamplesToLoad=10000):
    if not os.path.isdir(dataPath):
        os.makedirs(dataPath)
    startTime = time.time()
    
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'
    localDestination = os.path.join(dataPath, os.path.basename(url))

    download_dataset ( url, localDestination )
    
    columns = ['label','lepton_pT','lepton_eta','lepton_phi',
               'missing_energy_magnitude','missing_energy_phi','jet_1_pt',
               'jet_1_eta','jet_1_phi','jet_1_b_tag','jet_2_pt','jet_2_eta','jet_2_phi','jet_2_b_tag',
               'jet_3_pt','jet_3_eta','jet_3_phi','jet_3_b-tag','jet_4_pt','jet_4_eta','jet_4_phi',
               'jet_4_b_tag','m_jj','m_jjj','m_lv','m_jlv','m_bb','m_wbb','m_wwbb']
    
    print(f"reading HIGGS from local copy [ via pandas reader ]") # TODO fix cudf.read_csv ( compression = 'gzip')
    higgs = pd.read_csv( localDestination, nrows = nSamplesToLoad, header = None, names = columns  )

    data = cudf.DataFrame.from_pandas( higgs.iloc[:, 1:] )
    labels = cudf.DataFrame.from_pandas(higgs.iloc[:, 0].to_frame() )
    
    timeToLoad = time.time() - startTime
    return data, labels, timeToLoad

def load_airline_dataset (dataPath='./data/airline', nSamplesToLoad=10000):
    if not os.path.isdir(dataPath):
        os.makedirs(dataPath)

    startTime = time.time()
    
    url = 'http://kt.ijs.si/elena_ikonomovska/datasets/airline/airline_14col.data.bz2'
    localDestination = os.path.join( dataPath, os.path.basename(url))

    download_dataset ( url, localDestination )

    cols = [ "Year", "Month", "DayofMonth", "DayofWeek", "CRSDepTime",
             "CRSArrTime", "UniqueCarrier", "FlightNum", "ActualElapsedTime",
             "Origin", "Dest", "Distance", "Diverted", "ArrDelay" ]

    print(f"reading AIRLINE from local copy [ via pandas reader ]") # TODO fix cudf.read_csv ( compression = 'bzip?')
    df = pd.read_csv( localDestination, names = cols, nrows = nSamplesToLoad)

    # Encode categoricals as numeric
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].astype('category').cat.codes.astype( np.float32 )

    # Turn into binary classification problem
    df["ArrDelayBinary"] = 1. * (df["ArrDelay"] > 5)

    data = cudf.DataFrame.from_pandas( df[df.columns.difference(["ArrDelay", "ArrDelayBinary"])] )
    labels = cudf.DataFrame.from_pandas( df["ArrDelayBinary"].to_frame() )
    
    timeToLoad = time.time() - startTime
    return data, labels, timeToLoad

def load_fashion_mnist_dataset ( dataPath='./data/fmnist', nSamplesToLoad = 10000):
    startTime = time.time()
    trainDataURL = 'https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz'
    trainLabelsURL = 'https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz'
    if not os.path.isdir(dataPath):
        os.mkdirs(dataPath)
    localDestinationTrainData = os.path.join( dataPath, os.path.basename(trainDataURL))
    localDestinationTrainLabels = os.path.join( dataPath, os.path.basename(trainLabelsURL))

    download_dataset ( trainDataURL, localDestinationTrainData )
    download_dataset ( trainLabelsURL, localDestinationTrainLabels )

    with gzip.open(localDestinationTrainLabels, 'rb') as lbpath:
        labelsNP = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8, count=nSamplesToLoad)

    with gzip.open(localDestinationTrainData, 'rb') as imgpath:
        dataNP = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16, count=nSamplesToLoad * 784).reshape(len(labelsNP), 784)

    data = cudf.DataFrame.from_pandas( pd.DataFrame( dataNP.astype(np.float32) ) )

    labels = cudf.DataFrame.from_pandas( pd.DataFrame( labelsNP.astype(np.float32) ) )    

    timeToLoad = time.time() - startTime

    return data, labels, timeToLoad
    
def load_higgs_kaggle ( url = '!pip install kaggle --upgrade && export KAGGLE_USERNAME=kaggletracker123 && export KAGGLE_KEY=0fed3db9d2fb1ac0e430a29071953c5f && kaggle competitions download -c higgs-boson', 
                        localDestination = './data/higgs_kaggle_training.csv' ):
    startTime = time.time()
    datasetDF = pd.read_csv( localDestination )

    data = cudf.DataFrame.from_pandas( datasetDF.iloc[:,:-1] )
    labels = cudf.DataFrame.from_pandas( datasetDF['Label'].astype('category').cat.codes.to_frame() )

    data.shape, labels.shape
    timeToLoad = time.time() - startTime
    return data, labels, timeToLoad

''' -------------------------------------------------------------------------
>  DATA GEN [ CPU ]
------------------------------------------------------------------------- '''

def gen_coil ( nPoints= 100, coilType='helix',  coilDensity = 10, direction = 1 ):
    x = np.linspace( 0, coilDensity * np.pi, nPoints )    
    y = np.cos(x) * direction
    z = np.sin(x) * direction

    if coilType == 'whirl':
        y *= x*x/100
        z *= x*x/100
        
    points = np.vstack([x,y,z]).transpose()    
    return points

def gen_two_coils ( nPoints, coilType, coilDensity ):
    coil1 = gen_coil( nPoints, coilType, coilDensity, direction = 1)
    coil2 = gen_coil( nPoints, coilType, coilDensity, direction = -1)
    return coil1, coil2

def plot_dataset_variants ( nPoints = 100 ):
    global rapidsColors
    fig = plt.figure(figsize=(10,10))
        
    hCoil1, hCoil2 = gen_two_coils ( nPoints, coilType = 'helix', coilDensity = 5)
    bCoil1, bCoil2 = gen_two_coils ( nPoints, coilType = 'whirl', coilDensity = 4.25)
    
    # double helix
    ax = fig.add_subplot(211, projection='3d')
    ax.plot( bCoil1[:,0], bCoil1[:,1], bCoil1[:,2], 'gray', lw=1)
    ax.scatter( bCoil1[:,0], bCoil1[:,1], bCoil1[:,2], color =rapidsColors[0])
    ax.plot( bCoil2[:,0], bCoil2[:,1], bCoil2[:,2], 'gray', lw=1)
    ax.scatter( bCoil2[:,0], bCoil2[:,1], bCoil2[:,2], color = rapidsColors[1])

    ax2 = fig.add_subplot(212, projection='3d')    
    # whirl
    ax2.plot( hCoil1[:,0], hCoil1[:,1], hCoil1[:,2], 'gray', lw=1)
    ax2.scatter( hCoil1[:,0], hCoil1[:,1], hCoil1[:,2], color = rapidsColors[0])
    ax2.plot( hCoil2[:,0], hCoil2[:,1], hCoil2[:,2], 'gray', lw=1)
    ax2.scatter( hCoil2[:,0], hCoil2[:,1], hCoil2[:,2], color = rapidsColors[1])
    
    plt.show()
    
''' -------------------------------------------------------------------------
>  DATA GEN [ GPU ]
------------------------------------------------------------------------- '''

def generate_blobs ( nBlobPoints = 100,  coordinates = [],
                     sdevScales = [.05, .05, .05], noiseScale = 0.,
                     label = 0, shuffleFlag = False, rSeed = None ):
    ''' add blobs around list of input coordinates on the GPU [ using cupy.random.normal() ]
        --- inputs --- 
            nBlobPoints : number of points per blob            
            coordinates : list of starting locations of each blob
            sdevScales  : the standard deviation in each dimension
            noiseScale  : additional noise in the standard deviation, enable by setting to a positive value
            label       : the class value associated with all samples in this data
            rSeed       : random seed for repeatability
        --- returns --- 
            X : cuda DataFrame 3D points
            y : cuda DataFrame label
            t : elapsed time  
    '''
    if rSeed is not None:
        cupy.random.seed(rSeed)        
    
    nCoordinates = len(coordinates)
    print('generating blobs; # points = {}'.format(nBlobPoints * nCoordinates))
    startTime = time.time()    
    
    data = None
    labels = cupy.ones(( nBlobPoints * nCoordinates ), dtype='int32') * label
    
    # generate blobs [ one per coordinate ]
    for iCoordinate in range ( nCoordinates ):
        distributionCenter = cupy.array( coordinates[iCoordinate] )
        
        # generate blob using 3D unifrom normal distribution with ranomized standard deviation
        clusterData =  cupy.random.normal( loc = distributionCenter,
                                           scale = cupy.array(sdevScales) + cupy.random.randn() * noiseScale, 
                                           size = (nBlobPoints, 3) )

        if data is None:  data = clusterData
        else: data = cupy.concatenate((data, clusterData))            

    return data, labels, time.time() - startTime
    
def generate_dataset ( coilType = 'helix',
                       nSamples = 100000, 
                       coilDensity = 12,
                       sdevScales = [ .3, .3, .3], 
                       noiseScale = 1/10.,
                       shuffleFlag = False, 
                       rSeed = 0 ):
    
    startTime = time.time()
    
    nCoordinates = 400
    nBlobPoints = ( nSamples // 2 ) // nCoordinates
    
    coil1, coil2 = gen_two_coils ( nCoordinates, coilType = coilType, coilDensity = coilDensity )
    
    coil1_blobs, coil1_labels, t_gen1 = generate_blobs( coordinates = coil1, nBlobPoints = nBlobPoints, 
                                                               label = 0, 
                                                               noiseScale = noiseScale, sdevScales = sdevScales, 
                                                               shuffleFlag = shuffleFlag, rSeed = rSeed )
    
    coil2_blobs, coil2_labels, t_gen2 = generate_blobs( coordinates = coil2, nBlobPoints = nBlobPoints, 
                                                               label = 1, 
                                                               noiseScale = noiseScale, sdevScales = sdevScales, 
                                                               shuffleFlag = shuffleFlag, rSeed = rSeed )
    
    data = cupy.empty( shape = ( coil1_blobs.shape[0] + coil2_blobs.shape[0], coil1_blobs.shape[1] ),
                       dtype = coil1_blobs.dtype )
    labels = cupy.empty( shape = ( coil1_labels.shape[0] + coil2_labels.shape[0], ),
                       dtype = coil1_labels.dtype )

    # interleave coil1 and coil2 data and labels
    data[0::2] = coil1_blobs
    data[1::2] = coil2_blobs

    labels[0::2] = coil1_labels
    labels[1::2] = coil2_labels
    
    if shuffleFlag:
        shuffledInds = cupy.random.permutation (data.shape[0])
        data = data[shuffledInds]
        labels = labels[shuffledInds]
    
    data_cDF, labels_cDF = convert_to_cuDFs ( data, labels )
    elapsedTime = time.time() - startTime    
    return data_cDF, labels_cDF, elapsedTime

def convert_to_cuDFs ( data, labels ):
    '''  build cuda DataFrames for data and labels from cupy arrays '''
    labels_cDF = cudf.DataFrame([('labels', labels)])
    data_cDF = cudf.DataFrame([('x', cupy.asfortranarray(data[:,0])), 
                               ('y', cupy.asfortranarray(data[:,1])), 
                               ('z', cupy.asfortranarray(data[:,2]))])
    return data_cDF, labels_cDF

def ipv_plot_coils ( coil1, coil2, maxSamplesToPlot = 10000):
    global rapidsColors
    assert( type(coil1) == np.ndarray and type(coil2) == np.ndarray)    
    maxSamplesToPlot = min( ( coil1.shape[0], maxSamplesToPlot ) )
    stride = np.max((1, coil1.shape[0]//maxSamplesToPlot))
    
    print( '\t plotting data - stride = {} '.format( stride ) )
    ipv.figure()
    ipv.scatter( coil1[::stride,0], coil1[::stride,1], coil1[::stride,2], 
                size = .5, marker = 'sphere', color = rapidsColors[0])
    ipv.scatter( coil2[::stride,0], coil2[::stride,1], coil2[::stride,2], 
                size = .5, marker = 'sphere', color = rapidsColors[1])
    ipv.pylab.squarelim()
    ipv.show()
           
''' ---------------------------------------------------------------------
>  ETL - split and scale
----------------------------------------------------------------------'''
def split_train_test_nfolds ( dataDF, labelsDF, nFolds = 10, seed = 1, trainTestOverlap = .01 ):
    print('splitting data into training and test set')
    startTime = time.time()
    
    nSamplesPerFold = int(dataDF.shape[0] // nFolds)
    sampleRanges = np.arange(nFolds) * nSamplesPerFold
        
    np.random.seed(seed)
    foldStartInds = np.random.randint(0, nFolds-1, size = nFolds)
    foldEndInds = foldStartInds + 1 
    
    testFold = np.random.randint(0,nFolds-1)
    trainInds = None; testInds = None
    
    for iFold in range( nFolds ):
        lastFoldFlag = ( iFold == nFolds-1 )
        if lastFoldFlag: foldInds = np.arange(sampleRanges[iFold], dataDF.shape[0] )
        else: foldInds = np.arange(sampleRanges[iFold], sampleRanges[iFold+1])
        
        if iFold == testFold: testInds = foldInds
        else:
            if trainInds is None: trainInds = foldInds
            else: trainInds = np.concatenate([trainInds, foldInds])
                
    # swap subset of train and test samples [ low values require higher model generalization ]
    nSamplesToSwap = int( nSamplesPerFold * trainTestOverlap )
    if nSamplesToSwap > 0:
        trainIndsToSwap = np.random.permutation(trainInds.shape[0])[0:nSamplesToSwap]
        testIndsToSwap = np.random.permutation(testInds.shape[0])[0:nSamplesToSwap]        
        trainBuffer = trainInds[trainIndsToSwap].copy()
        trainInds[trainIndsToSwap] = testInds[testIndsToSwap]
        testInds[testIndsToSwap] = trainBuffer
    
    # build final dataframes    
    trainDF = dataDF.iloc[trainInds]
    testDF = dataDF.iloc[testInds]
    trainLabelsDF = labelsDF.iloc[trainInds]
    testLabelsDF = labelsDF.iloc[testInds]
    return trainDF, trainLabelsDF, testDF, testLabelsDF, time.time() - startTime

def scale_dataframe_inplace ( targetDF, trainMeans = {}, trainSTDevs = {} ):    
    print('rescaling data')
    sT = time.time()
    for iCol in targetDF.columns:
        
        # omit scaling label column
        if iCol == targetDF.columns[-1] == 'label': continue
            
        # compute means and standard deviations for each column [ should skip for test data ]
        if iCol not in trainMeans.keys() and iCol not in trainSTDevs.keys():            
            trainMeans[iCol] = targetDF[iCol].mean()
            trainSTDevs[iCol] = targetDF[iCol].std()
            
        # apply scaling to each column
        targetDF[iCol] = ( targetDF[iCol] - trainMeans[iCol] ) / ( trainSTDevs[iCol] + 1e-10 )
        
    return trainMeans, trainSTDevs, time.time() - sT

''' -------------------------------------------------------------------------
>  VISUALIZE TRAIN & TEST + OVERLAP
------------------------------------------------------------------------- '''

def plot_train_test ( trainData, trainLabels, testData, testLabels, maxSamplesToPlot = 10000):    
    global rapidsColors
    
    minStride = 2
    trainStride = trainData.shape[0] // maxSamplesToPlot
    if trainStride % minStride != 0:
        trainStride += 1
    testStride = testData.shape[0] // maxSamplesToPlot
    if testStride % minStride != 0:
        testStride += 1
        
    strideStepTrain = np.max((minStride, trainStride))
    strideStepTest = np.max((minStride, testStride))

    coil1TrainData = trainData[0::strideStepTrain].to_pandas().values.astype( 'float64')
    coil2TrainData = trainData[1::strideStepTrain].to_pandas().values.astype( 'float64')
    coil1TestData = testData[0::strideStepTest].to_pandas().values.astype( 'float64')
    coil2TestData = testData[1::strideStepTest].to_pandas().values.astype( 'float64')

    ipv.figure()
    
    # train data
    ipv.scatter(coil1TrainData[:,0], coil1TrainData[:,1], coil1TrainData[:,2], 
                size = .25, color=rapidsColors[0], marker='sphere')
    ipv.scatter(coil2TrainData[:,0], coil2TrainData[:,1], coil2TrainData[:,2], 
                size = .25, color=rapidsColors[1], marker='sphere')

    # test data callout
    offset = np.max((coil1TrainData[:,2])) * 3
    ipv.scatter( coil1TestData[:,0], coil1TestData[:,1], coil1TestData[:,2] + offset, 
                 size = .25, color=rapidsColors[0], marker='sphere')
    ipv.scatter( coil2TestData[:,0], coil2TestData[:,1], coil2TestData[:,2] + offset, 
                 size = .25, color=rapidsColors[1], marker='sphere')
    
    ipv.pylab.squarelim()
    ipv.show()
    

def plot_iid_breaking ( trainData_pDF, testData_pDF, maxSamplesToPlot = 10000 ):
    global rapidsColors
    plt.figure(figsize=(50,10))
    plt.subplot(1,3,1)
    trainStride = np.max( (1, trainData_pDF.shape[0] // maxSamplesToPlot) )
    testStride = np.max( (1, testData_pDF.shape[0] // maxSamplesToPlot) )
    rapidsColors[1] = [ 0, 0, 0, 1]
    rapidsColors[3] = [ 102/255., 102/255., 102/255., 1]
    print( f'train data stride {trainStride}, test data stride {testStride}')
    plt.plot(testData_pDF['x'].iloc[::testStride],'o', color = rapidsColors[1] )
    plt.plot(trainData_pDF['x'].iloc[::trainStride],'x', color = rapidsColors[3])
    plt.legend(['testData', 'trainData'])
    plt.title('x')
    
    plt.subplot(1,3,2)
    plt.plot(testData_pDF['y'].iloc[::testStride],'o', color = rapidsColors[1] )
    plt.plot(trainData_pDF['y'].iloc[::trainStride],'x', color = rapidsColors[3])
    plt.legend(['testData', 'trainData'])
    plt.title('y')
    
    plt.subplot(1,3,3)
    plt.plot(testData_pDF['z'].iloc[::testStride],'o', color = rapidsColors[1] )
    plt.plot(trainData_pDF['z'].iloc[::trainStride],'x', color = rapidsColors[3])
    plt.legend(['testData', 'trainData'])
    plt.title('z')
    plt.show()   

