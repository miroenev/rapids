import numpy as np
import pandas as pd

import cupy
import cudf
import cuml

import time
import gzip 

import os
import warnings

from urllib.request import urlretrieve

class Dataset():    
    def __init__( self, datasetName = 'synthetic', nSamples = None, 
                  coilType = 'helix', coilDensity = 15, 
                  coil1StDev = .3, coil2StDev = .3, 
                  nGuidePointCenters = 1000, 
                  randomSeed = 0, shuffleFlag = True ):

        self.datasetName = datasetName
        
        if self.datasetName == 'synthetic':
            self.coilType = coilType
            if nSamples == None: nSamples = 1000000
            self.data, self.labels, elapsedTime  = generate_synthetic_dataset( coilType = coilType, nSamples = nSamples,
                                                                               coilDensity = coilDensity,
                                                                               coil1StDev = coil1StDev, coil2StDev = coil2StDev,
                                                                               nGuidePointCenters = nGuidePointCenters, 
                                                                               shuffleFlag = shuffleFlag )
            self.trainObjective = ['binary:hinge', None]
            
        elif self.datasetName == 'fashion-mnist':

            self.data, self.labels, elapsedTime = load_fashion_mnist_dataset ( './data/fashion-mnist' ) 
            self.trainObjective = ['multi:softmax', 10]

        elif self.datasetName == 'airline':
            
            if nSamples == None: nSamples = 5000000
            self.data, self.labels, elapsedTime = load_airline_dataset ( './data/airline', np.min( ( nSamples, 115000000 )))
            self.trainObjective = ['binary:hinge', None]
            
        self.trainData = self.trainLabels = self.testData = self.testLabels = None
        
        self.samplesToSwap = int(self.data.shape[0] * .002)     # samples to exchange between train and test set [ enables generalization in the synthetic data case ]
        self.percentTrain = .885                                # precent of the dataset to use for training 
        self.nScalingRuns = 0                                   # book-keeping guard to track number of times dataset has been rescaled
        
    def assign_dataset_splits ( self, trainData, testData, trainLabels, testLabels ):
        self.trainData = trainData
        self.trainLabels = trainLabels
        self.testData = testData
        self.testLabels = testLabels
        self.nScalingRuns = 0
        
''' -------------------------------------------------------------------------
>  public dataset loading 
------------------------------------------------------------------------- '''
def data_progress_hook ( blockNumber, readSize, totalFileSize ):
    if ( blockNumber % 1000 ) == 0:        
        print(f' > percent complete: { 100 * ( blockNumber * readSize ) / totalFileSize:.2f}\r', end='')
    return

def download_dataset ( url, localDestination, reportHook = None ):
    if not os.path.isfile( localDestination ):
        print(f'no local dataset copy at {localDestination}')
        print(f' > downloading dataset from: {url}')
        urlretrieve( url = url, filename = localDestination, reporthook = reportHook )

def load_airline_dataset (dataPath='./data/airline', nSamplesToLoad=10000):
    if not os.path.isdir(dataPath):
        os.makedirs(dataPath)

    startTime = time.time()
    
    url = 'http://kt.ijs.si/elena_ikonomovska/datasets/airline/airline_14col.data.bz2'
    localDestination = os.path.join( dataPath, os.path.basename(url))

    download_dataset ( url, localDestination, data_progress_hook )

    cols = [ "Year", "Month", "DayofMonth", "DayofWeek", "CRSDepTime",
             "CRSArrTime", "UniqueCarrier", "FlightNum", "ActualElapsedTime",
             "Origin", "Dest", "Distance", "Diverted", "ArrDelay" ]

    print(f"reading AIRLINE from local copy [ via pandas reader ]") # TODO fix cudf.read_csv ( compression = 'bzip?')
    df = pd.read_csv( localDestination, names = cols, nrows = nSamplesToLoad)

    # encode categoricals as numeric
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].astype('category').cat.codes.astype( np.int32 )
    
    # cast all remaining columns to int32
    for col in df.columns:
        if col in df.select_dtypes(['object']).columns: pass
        else:
            df[col] = df[col].astype( np.int32 )
        
    # turn into binary classification problem [i.e. flights delayed beyond 5 minutes are considered late ]
    df["ArrDelayBinary"] = 1. * (df["ArrDelay"] > 5)

    data = cudf.DataFrame.from_pandas( df[df.columns.difference(["ArrDelay", "ArrDelayBinary"])] )
    labels = cudf.DataFrame.from_pandas( df["ArrDelayBinary"].to_frame() )
    
    timeToLoad = time.time() - startTime
    return data, labels, timeToLoad

def load_fashion_mnist_dataset ( dataPath='./data/fmnist', nSamplesToLoad = 60000):
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
    
''' -------------------------------------------------------------------------
>  synthetic data generation
------------------------------------------------------------------------- '''
# generate guide-points / centers along a coil onto which random blob points will be sampled/generated
def gen_coil ( nPoints= 100, coilType='helix',  coilDensity = 10, coilDiameter = 3, direction = 1 ):
    x = np.linspace( 0, coilDensity * np.pi, nPoints )    
    y = coilDiameter * np.cos( x ) * direction
    z = coilDiameter * np.sin( x ) * direction

    if coilType == 'whirl':        
        y /= float(coilDiameter)
        z /= float(coilDiameter)
        y *= x*x/( coilDensity * 8 )
        z *= x*x/( coilDensity * 8 )        
        
    points = np.vstack([x,y,z]).transpose()    
    return points

def gen_two_coils ( nPoints, coilType, coilDensity ):
    coil1 = gen_coil( nPoints, coilType, coilDensity, direction = 1)
    coil2 = gen_coil( nPoints, coilType, coilDensity, direction = -1)
    return coil1, coil2

def generate_synthetic_dataset ( coilType = 'helix', nSamples = 1000000, coilDensity = 12,
                                 coil1StDev = .3, coil2StDev = .3,
                                 nGuidePointCenters = 500,
                                 randomSeed = 0, shuffleFlag = True ):
    
    startTime = time.time()
    
    coil1Centers, coil2Centers = gen_two_coils( nPoints = nGuidePointCenters, 
                                                           coilType = coilType, 
                                                           coilDensity = coilDensity )
    
    samplesPerCoil = nSamples // 2 
    nDims = 3
    
    coil1Data, _ = cuml.make_blobs ( n_samples = samplesPerCoil, n_features=nDims, centers=coil1Centers, 
                                     cluster_std = coil1StDev, random_state = 0, dtype = 'float')

    coil2Data, _ = cuml.make_blobs ( n_samples = samplesPerCoil, n_features=nDims, centers=coil2Centers, 
                                     cluster_std = coil2StDev, random_state = 0, dtype = 'float')

    combinedData = cupy.empty( shape = (samplesPerCoil * 2, nDims), dtype = 'float32', order='F')
    combinedData[0::2] = coil1Data
    combinedData[1::2] = coil2Data

    combinedLabels = cupy.empty( shape = (samplesPerCoil * 2, 1), dtype = 'int', order='F')
    combinedLabels[0::2] = cupy.ones( shape = (samplesPerCoil, 1), dtype = 'int' )
    combinedLabels[1::2] = cupy.zeros( shape = (samplesPerCoil, 1), dtype = 'int' )
    
    if shuffleFlag:
        cupy.random.seed(0)
        shuffledInds = cupy.random.permutation (combinedData.shape[0])
        combinedData = cupy.asfortranarray(combinedData[shuffledInds, :])
        combinedLabels = cupy.asfortranarray(combinedLabels[shuffledInds])

    data = cudf.DataFrame.from_gpu_matrix( combinedData, columns = ['x', 'y', 'z'] )
    labels = cudf.DataFrame.from_gpu_matrix( combinedLabels, columns = ['labels'] )        
    
    elapsedTime = time.time() - startTime
    return data, labels, elapsedTime
           
''' ---------------------------------------------------------------------
>  ETL - split and scale
----------------------------------------------------------------------'''
# extract the middle of the dataset and assign it as test data
# swap a subset of samples between the train and test set to help generalization to the test region
def split_synthetic ( data, labels, percentTrain = .8, 
                      samplesToSwap = 2000, targetDim ='x' ):
    
    #self.samplesToSwap = int(self.data.shape[0] * .002)     # samples to exchange between train and test set [ enables generalization in the synthetic data case ]
    #self.percentTrain = .885                                # precent of the dataset to use for training 
    
    
    print(f"splitting synthetic dataset into train-set {percentTrain*100}% and test-set {round((1-percentTrain)*100,10)}%")
    dataSpan = ( data[targetDim].max() - data[targetDim].min() )
    dataMiddle = dataSpan * .5
    testDataSpan = dataSpan*(1-percentTrain)
    
    print(f'> assigning middle span of data to test-set')
    testData = data[data[targetDim].ge( dataMiddle - testDataSpan/2. )]
    testData = testData[testData[targetDim].le( dataMiddle + testDataSpan/2. )]
    trainData = data[data[targetDim].ne( testData[targetDim]) ]
    testLabels = labels.iloc[testData.index]
    trainLabels = labels.iloc[trainData.index]

    #samplesToSwap = int( trainData.shape[0] * trainTestExchange )
    print(f'> swapping {samplesToSwap} samples between train-set and test-set')#, i.e., {round(trainTestExchange*100,10)}% of train-set samples')
    if samplesToSwap > 0:
        # swap data
        for iColumn in trainData.columns:
            tempBuffer = trainData[iColumn][0:samplesToSwap].copy()
            trainData[iColumn][0:samplesToSwap] = testData[iColumn][0:samplesToSwap] 
            testData[iColumn][0:samplesToSwap] = tempBuffer
            
        # swap labels
        assert(labels.columns[0] == 'labels')
        tempBuffer = trainLabels['labels'][0:samplesToSwap].copy()
        trainLabels['labels'][0:samplesToSwap] = testLabels['labels'][0:samplesToSwap] 
        testLabels['labels'][0:samplesToSwap] = tempBuffer            

    return trainData, testData, trainLabels, testLabels  

def scale_dataframe_inplace ( targetDF, trainMeans = {}, trainSTDevs = {}, label='train-set', datasetObject = None):        
    
    # protect from multiple re-scaling attempts
    if datasetObject is not None:
        if datasetObject.nScalingRuns > 1: 
            print(f'The {label.upper()} dataset seems to already have been scaled! Be careful to only run scaling once per dataset.')            
            if input(f"Do you wish to scale {label.upper()} again? ( 'yes' / 'no' ) suggested = 'no': ",).lower() == 'no': 
                return None, None, None
        datasetObject.nScalingRuns += 1
        
    print(f'applying [inplace] standard scaling to {label} data')
    sT = time.time()
    for iCol in targetDF.columns:
        
        # compute means and standard deviations for each column [ should skip for test data ]
        if iCol not in trainMeans.keys() and iCol not in trainSTDevs.keys():            
            trainMeans[iCol] = targetDF[iCol].mean()
            trainSTDevs[iCol] = targetDF[iCol].std()
            
        # apply scaling to each column
        targetDF[iCol] = ( targetDF[iCol] - trainMeans[iCol] ) / ( trainSTDevs[iCol] + 1e-10 )
        
    return trainMeans, trainSTDevs, time.time() - sT

def print_stats ( data ):
    print(f"{'x':>20}{'y':>9}{'z':>9}")
    print(f'  train means  :{data.trainData.x.mean():>8.3f} {data.trainData.y.mean():>8.3f} {data.trainData.z.mean():>8.3f}')
    print(f'  train stDevs :{data.trainData.x.std():>8.3f} {data.trainData.y.std():>8.3f} {data.trainData.z.std():>8.3f}')
    print(f'\n  test  means  :{data.testData.x.mean():>8.3f} {data.testData.y.mean():>8.3f} {data.testData.z.mean():>8.3f}')
    print(f'  test  stDevs :{data.testData.x.std():>8.3f} {data.testData.y.std():>8.3f} {data.testData.z.std():>8.3f}')