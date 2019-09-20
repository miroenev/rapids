import numpy as np; import numpy.matlib
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipyvolume as ipv
import time
import cupy
import cudf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets; from sklearn.metrics import confusion_matrix, accuracy_score

import dask
from dask import delayed
import xgboost

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
    fig = plt.figure(figsize=(10,10))
        
    hCoil1, hCoil2 = gen_two_coils ( nPoints, coilType = 'helix', coilDensity = 5)
    bCoil1, bCoil2 = gen_two_coils ( nPoints, coilType = 'whirl', coilDensity = 4.25)
    
    # double helix
    ax = fig.add_subplot(211, projection='3d')
    ax.plot( bCoil1[:,0], bCoil1[:,1], bCoil1[:,2], 'gray', lw=1)
    ax.scatter( bCoil1[:,0], bCoil1[:,1], bCoil1[:,2], 'lightgreen')
    ax.plot( bCoil2[:,0], bCoil2[:,1], bCoil2[:,2], 'gray', lw=1)
    ax.scatter( bCoil2[:,0], bCoil2[:,1], bCoil2[:,2], color='purple')

    ax2 = fig.add_subplot(212, projection='3d')    
    # whirl
    ax2.plot( hCoil1[:,0], hCoil1[:,1], hCoil1[:,2], 'gray', lw=1)
    ax2.scatter( hCoil1[:,0], hCoil1[:,1], hCoil1[:,2], 'lightgreen')
    ax2.plot( hCoil2[:,0], hCoil2[:,1], hCoil2[:,2], 'gray', lw=1)
    ax2.scatter( hCoil2[:,0], hCoil2[:,1], hCoil2[:,2], color='purple')
    
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

def ipv_plot_coils ( coil1, coil2, maxSamplesToPlot = 10000):
    assert( type(coil1) == np.ndarray and type(coil2) == np.ndarray)    
    maxSamplesToPlot = min( ( coil1.shape[0], maxSamplesToPlot ) )
    stride = np.max((1, coil1.shape[0]//maxSamplesToPlot))
    
    print( '\t plotting data - stride = {} '.format( stride ) )
    ipv.figure()
    ipv.scatter( coil1[::stride,0], coil1[::stride,1], coil1[::stride,2], size = .5, marker = 'sphere', color = 'lightgreen')
    ipv.scatter( coil2[::stride,0], coil2[::stride,1], coil2[::stride,2], size = .5, marker = 'sphere', color = 'purple')
    ipv.pylab.squarelim()
    ipv.show()
    
    
def plot_train_test ( trainData, trainLabels, testData, testLabels, maxSamplesToPlot = 10000):    
    
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
    ipv.scatter(coil1TrainData[:,0], coil1TrainData[:,1], coil1TrainData[:,2], size = .25, color='lightgreen', marker='sphere')
    ipv.scatter(coil2TrainData[:,0], coil2TrainData[:,1], coil2TrainData[:,2], size = .25, color='purple', marker='sphere')

    # test data overlay on train data 
    #ipv.scatter(coil1TestData[:,0], coil1TestData[:,1], coil1TestData[:,2], size = .1, color='lightgray', marker='sphere')
    #ipv.scatter(coil2TestData[:,0], coil2TestData[:,1], coil2TestData[:,2], size = .1, color='lightgray', marker='sphere')

    # test data callout
    offset = np.max((coil1TrainData[:,2])) * 3
    ipv.scatter(coil1TestData[:,0], coil1TestData[:,1], coil1TestData[:,2] + offset, size = .25, color='lightgreen', marker='sphere')
    ipv.scatter(coil2TestData[:,0], coil2TestData[:,1], coil2TestData[:,2] + offset, size = .25, color='purple', marker='sphere')
    
    ipv.pylab.squarelim()
    ipv.show()
    
    
def gen_blob_coils ( nBlobPoints = 1000, 
                     nCoordinates = 400, 
                     coilType='whirl', 
                     coilDensity = 6.25, 
                     sdevScales = [ .05, .05, .05], 
                     noiseScale = 1/5., 
                     shuffleFlag = False, 
                     rSeed = 0, 
                     plotFlag = True,
                     maxSamplesToPlot = 50000):
    
    startTime = time.time()
    
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
    print( 'time to generate data on GPU = {}'.format( elapsedTime ) )
    
    if plotFlag:        
        ipv_plot_coils( cupy.asnumpy(coil1_blobs), cupy.asnumpy(coil2_blobs), maxSamplesToPlot = maxSamplesToPlot )
    
    return data_cDF, labels_cDF, elapsedTime

def convert_to_cuDFs ( data, labels ): # consider dlpack
    '''  build cuda DataFrames for data and labels from cupy arrays '''
    labels_cDF = cudf.DataFrame([('labels', labels)])
    data_cDF = cudf.DataFrame([('x', cupy.asfortranarray(data[:,0])), 
                               ('y', cupy.asfortranarray(data[:,1])), 
                               ('z', cupy.asfortranarray(data[:,2]))])
    return data_cDF, labels_cDF


''' ---------------------------------------------------------------------
>  MEASUREMENT / LOGGING
----------------------------------------------------------------------'''
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

''' ---------------------------------------------------------------------
>  HPO / PARTICLE SWARM
----------------------------------------------------------------------'''
def initalize_hpo ( nTimesteps, nParticles, nWorkers, paramRanges, nParameters = 3, plotFlag = True):
    accuracies = np.zeros( ( nTimesteps, nParticles) )
    bestParticleIndex = {}

    globalBestParticleParams = np.zeros((nTimesteps, 1, nParameters))
    particles = np.zeros( (nTimesteps, nParticles, nParameters ) )
    velocities = np.zeros( (nTimesteps, nParticles, nParameters ) )
    particleBoostingRounds = np.zeros((nTimesteps, nParticles))
    
    # initial velocity is one
    velocities[0, :, :] = np.ones( ( nParticles, nParameters ) ) * .25
    
    # randomly initialize particle colors
    particleColors = np.random.uniform( size = (1, nParticles, 3) )

    # best initialized to middle [ fictional particle ] -- is this necessary
    bestParticleIndex[0] = -1
        
    # grid initialize particles
    x = np.linspace( paramRanges[0][1], paramRanges[0][2], nWorkers)
    y = np.linspace( paramRanges[1][1], paramRanges[1][2], nWorkers)
    z = np.linspace( paramRanges[2][1], paramRanges[2][2], nWorkers)

    xx, yy, zz = np.meshgrid(x,y,z, indexing='xy')

    xS = xx.reshape(1,-1)[0]
    yS = yy.reshape(1,-1)[0]
    zS = zz.reshape(1,-1)[0]
    
    # clip middle particles
    particles[0, :, 0] = np.hstack([xS[-nWorkers**2:], xS[:nWorkers**2]])
    particles[0, :, 1] = np.hstack([yS[-nWorkers**2:], yS[:nWorkers**2]])
    particles[0, :, 2] = np.hstack([zS[-nWorkers**2:], zS[:nWorkers**2]])
    
    if plotFlag:
        ipv.figure()
        ipv.scatter( particles[0,:,0], particles[0,:,1], particles[0,:,2], marker = 'sphere', color=particleColors)
        ipv.xlim( paramRanges[0][1], paramRanges[0][2] )
        ipv.ylim( paramRanges[1][1], paramRanges[1][2] )
        ipv.zlim( paramRanges[2][1], paramRanges[2][2] )
        ipv.show()
        
    return particles, velocities, accuracies, bestParticleIndex, globalBestParticleParams, particleBoostingRounds, particleColors


def update_particles( paramRanges, particlesInTimestep, velocitiesInTimestep, bestParamsIndex, globalBestParams, sBest = .75, sExplore = .25 , deltaTime = 1, randomSeed = None):
    
    nParticles = particlesInTimestep.shape[ 0 ]
    nParameters = particlesInTimestep.shape[ 1 ]    
        
    globalBestRepeated = numpy.matlib.repmat( np.array( globalBestParams ).reshape( -1, 1 ), nParticles, 1).reshape( nParticles, nParameters )    
    
    if randomSeed is not None:
        np.random.seed(randomSeed)
        
    # move to best + explore | globalBest + personalBest
    velocitiesInTimestep += sBest * ( globalBestRepeated - particlesInTimestep ) \
                            + sExplore * ( np.random.randn( nParticles, nParameters ) )
    
    particlesInTimestep += velocitiesInTimestep * deltaTime 
    
    # TODO: avoid duplicates
    
    # enforce param bounds
    for iParam in range( nParameters ):
        particlesInTimestep[ :, iParam ] = np.clip(particlesInTimestep[ :, iParam ], paramRanges[iParam][1], paramRanges[iParam][2])
        if paramRanges[iParam][3] == 'int':
            particlesInTimestep[ :, iParam ] = np.round( particlesInTimestep[ :, iParam ] )
            
    return particlesInTimestep, velocitiesInTimestep
    
def train_model_hpo ( trainData_cDF, trainLabels_cDF, testData_cDF, testLabels_cDF, particleParams, iParticle, iTimestep ):
    
    # fixed parameters
    paramsGPU = { 'objective': 'binary:hinge',
                  'tree_method': 'gpu_hist',
                  'n_gpus': 1,
                  'random_state': 0 }
    
    # parameters to search over
    paramsGPU['max_depth'] = int(particleParams[0])
    paramsGPU['learning_rate'] = particleParams[1]
    paramsGPU['lambda'] = particleParams[2] 
    paramsGPU['num_boost_rounds'] = 1000
    
    startTime = time.time()
    trainDMatrix = xgboost.DMatrix( data = trainData_cDF, label = trainLabels_cDF )
    testDMatrix = xgboost.DMatrix( data = testData_cDF, label = testLabels_cDF )
    trainedModelGPU = xgboost.train( dtrain = trainDMatrix, evals = [(testDMatrix, 'test')], 
                                     params = paramsGPU,
                                     num_boost_round = paramsGPU['num_boost_rounds'],
                                     early_stopping_rounds = 15,
                                     verbose_eval = False )
    
    elapsedTime = time.time() - startTime
    print('training xgboost model on GPU t: {}, nP: {}, params [  {} {} {} ], time: {} '.format( iTimestep, iParticle, particleParams[0], particleParams[1], particleParams[2], elapsedTime) );  
    return trainedModelGPU, elapsedTime

def test_model_hpo ( trainedModelGPU, trainingTime, testData_cDF, testLabels_cDF ):
    
    startTime = time.time()
    
    testDMatrix = xgboost.DMatrix( data = testData_cDF, label = testLabels_cDF )    
    predictionsGPU = trainedModelGPU.predict( testDMatrix ).astype(int)
    
    return predictionsGPU, trainedModelGPU.best_iteration, trainingTime, time.time() - startTime


def run_hpo ( daskClient, nTimesteps, nParticles, nWorkers, paramRanges, trainData_cDF, trainLabels_cDF, testData_cDF, testLabels_cDF, randomSeed = 0):
    
    pandasTestLabels = testLabels_cDF.to_pandas()

    if daskClient is not None:        
        scatteredData_future = daskClient.scatter( [ trainData_cDF, trainLabels_cDF, testData_cDF, testLabels_cDF ], broadcast = True )
    
    trainData_cDF_future = scatteredData_future[0]; trainLabels_cDF_future = scatteredData_future[1]
    testData_cDF_future = scatteredData_future[2]; testLabels_cDF_future = scatteredData_future[3]
    
    particles, velocities, accuracies, bestParticleIndex, \
        globalBestParticleParams, particleBoostingRounds, particleColors = initalize_hpo ( nTimesteps = nTimesteps, 
                                                                                           nParticles = nParticles, 
                                                                                           nWorkers = nWorkers, 
                                                                                           paramRanges = paramRanges)
    globalBestAccuracy = 0
    
    trainingTimes = np.zeros (( nTimesteps, nParticles ))    
    startTime = time.time()
    
    predictionHistory = np.zeros((nTimesteps, nParticles, testData_cDF.shape[0]))
    
    for iTimestep in range (0, nTimesteps ):    
        if daskClient is not None:
            # [ delayed ] train xgboost models on train data
            delayedParticleTrain = [ delayed( train_model_hpo )( trainData_cDF_future, trainLabels_cDF_future, 
                                                                     testData_cDF_future, testLabels_cDF_future, 
                                                                     particles[iTimestep, iParticle, : ], 
                                                                         iParticle, iTimestep) for iParticle in range(nParticles) ]

            # [ delayed ] determine number of trees/training-rounds returned early stopping -- used to set particle sizes
            delayedParticleRounds = [ iParticle[0].best_iteration for iParticle in delayedParticleTrain ]
            
            # [delayed ] eval trained models on test/validation data
            delayedParticlePredictions = [ delayed( test_model_hpo )(iParticle[0], iParticle[1], 
                                                                     testData_cDF_future, 
                                                                     testLabels_cDF_future) for iParticle in delayedParticleTrain ]
            
            # execute delayed             
            particlePredictions = dask.compute( delayedParticlePredictions )[0]            
            
            
            for iParticle in range(nParticles):
                predictionHistory[iTimestep, iParticle, :] = particlePredictions[iParticle][0]
            #import pdb; pdb.set_trace()
            
            # compute accuracies of predictions
            accuracies[iTimestep, :] = [ accuracy_score ( pandasTestLabels, iParticle[0]) for iParticle in particlePredictions ]
            particleBoostingRounds[iTimestep, : ] = [ iParticle[1] for iParticle in particlePredictions ]
            trainingTimes[iTimestep, :] = [ iParticle[2] for iParticle in particlePredictions ]
            del particlePredictions
        else:
            for iParticle in range(nParticles):
                trainedModels, _ = train_model_hpo ( pandasTrainData, pandasTrainLabels, particles[iTimestep, iParticle, : ], iParticle, iTimestep)
                predictions, _ = test_model_hpo( trainedModels, pandasTestData, pandasTestLabels)            
                accuracies[iTimestep, iParticle] = accuracy_score (pandasTestLabels, predictions)
        
        bestParticleIndex[iTimestep+1] = np.argmax( accuracies[iTimestep, :] )
        currentBestAccuracy = np.max( accuracies[iTimestep, :] )

        print('@ hpo timestep : {}, best accuracy is {}'.format(iTimestep, np.max(accuracies[iTimestep, :])) )
        if iTimestep +1 < nTimesteps:            
            if currentBestAccuracy > globalBestAccuracy:
                print('\t updating best GLOBAL accuracy')
                globalBestAccuracy = currentBestAccuracy
                globalBestParticleParams[iTimestep+1] = particles[iTimestep, bestParticleIndex[iTimestep+1], :]
            else:
                globalBestParticleParams[iTimestep+1] = globalBestParticleParams[iTimestep].copy()
            
            particles[iTimestep+1, :, :], velocities[iTimestep+1, :, : ] = update_particles ( paramRanges, 
                                                                                              particles[iTimestep, :, :].copy(),
                                                                                              velocities[iTimestep, :, :].copy(), 
                                                                                              bestParticleIndex[iTimestep+1], 
                                                                                              globalBestParticleParams[iTimestep+1], randomSeed = iTimestep)
    
    particleSizes = particleBoostingRounds/np.max(particleBoostingRounds)*10 + 2
    elapsedTime = time.time() - startTime
    print( 'elapsed time : {}'.format(elapsedTime) )
    
    return accuracies, particles, velocities, particleSizes, particleColors, bestParticleIndex, particleBoostingRounds, trainingTimes, predictionHistory, elapsedTime

def viz_search( accuracies, particleBoostingRounds ):
    fig = plt.figure(figsize=(30,20))
    plt.subplot(1,2,1)
    plt.title('accuracies')
    plt.imshow(accuracies, cmap='jet')
    plt.colorbar(shrink=.1)
    plt.xlabel('particle'); plt.ylabel('timestep')
    plt.subplot(1,2,2)
    plt.title('boosting rounds')
    plt.imshow(particleBoostingRounds, cmap='jet')
    plt.colorbar(shrink=.1)
    plt.xlabel('particle'); plt.ylabel('timestep')
    plt.show()
    
def hpo_animate ( particles, particleSizes, particleColors, paramRanges, nTimesteps = 1 ):
    nParticles = particles.shape[1]
    colorStack = np.ones((nTimesteps, nTimesteps * nParticles,4)) * .9
    colorStackLines = np.ones((nTimesteps, nTimesteps * nParticles,4)) * .5
    for iTimestep in range(nTimesteps):    
        colorStack[iTimestep, :, 0:3] = numpy.matlib.repmat( particleColors[0,:,:], nTimesteps, 1)
        colorStackLines[iTimestep, :, 0:3] = numpy.matlib.repmat( particleColors[0,:,:], nTimesteps, 1)
        colorStackLines[iTimestep, :, 3] = .6 # alpha
    ipv.figure()

    pplot = ipv.scatter( particles[0:nTimesteps,:,0],
                         particles[0:nTimesteps,:,1],
                         particles[0:nTimesteps,:,2], marker='sphere', size=particleSizes, color=colorStack[:,:,:])

    for iParticle in range(nParticles):
        plines = ipv.plot(particles[0:nTimesteps,iParticle,0], 
                          particles[0:nTimesteps,iParticle,1], 
                          particles[0:nTimesteps,iParticle,2], color=colorStackLines[:,iParticle,:] )

    ipv.animation_control( [ pplot ] , interval=600 )
    ipv.xlim( paramRanges[0][1]-.5, paramRanges[0][2]+.5 )
    ipv.ylim( paramRanges[1][1]-.1, paramRanges[1][2]+.1 )
    ipv.zlim( paramRanges[2][1]-.1, paramRanges[2][2]+.1 )
    ipv.show()    
    
    
def plot_particle_learning ( nTimesteps, nParticles, testData_pDF, bestParamIndex, predictionHistory ):
    
    nTestSamples = testData_pDF.shape[0]
    colorStack = np.ones((nTimesteps, nParticles, nTestSamples, 4)) * [ 1, 0, 0, 1 ]
    for iTimestep in range( nTimesteps ):
        for iParticle in range( nParticles):
            colorStack[iTimestep, iParticle, :, 0] = predictionHistory[iTimestep, iParticle, :]    
    
    nTestSamples = testData_pDF.shape[0]
    testDataRepeated = np.zeros((nTimesteps, nParticles, nTestSamples, 3))
    xNP = testData_pDF['x'].values
    yNP = testData_pDF['y'].values
    zNP = testData_pDF['z'].values
    for iTimestep in range( nTimesteps ):
        for iParticle in range( nParticles ):
            testDataRepeated[iTimestep, iParticle, :, 0] = xNP
            testDataRepeated[iTimestep, iParticle, :, 1] = yNP
            testDataRepeated[iTimestep, iParticle, :, 2] = zNP

    ipv.figure()
    nTimestepsToPlot = bestParamIndex[0]+1
    bestParticle = bestParamIndex[1]

    predictionPlots = ipv.scatter( testDataRepeated[0:nTimestepsToPlot, bestParticle, :, 0], 
                                   testDataRepeated[0:nTimestepsToPlot, bestParticle, :, 1], 
                                   testDataRepeated[0:nTimestepsToPlot, bestParticle, :, 2],
                                   color = colorStack[0:nTimestepsToPlot, bestParticle, :, :], marker='sphere', size=.25 )

    ipv.animation_control( [ predictionPlots ] , interval=600)
    ipv.pylab.squarelim()
    ipv.show()
    
    '''
    [e.g. single particle learning ]
    ipv.figure()
    ipv.scatter( testData_pDF['x'].values, testData_pDF['y'].values, testData_pDF['z'].values,
                               color = colorStack[0, 0, :, :], marker='sphere', size=.25 )
    ipv.show()
    '''
       
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

def plot_iid_breaking ( trainData_cDF, testData_cDF ):
    plt.figure(figsize=(50,10))
    plt.subplot(1,3,1)
    subSample = 100
    plt.plot(testData_cDF['x'].iloc[::subSample],'o', color=[1,0,0,.5] )
    plt.plot(trainData_cDF['x'].iloc[::subSample],'x')
    plt.legend(['testData', 'trainData'])
    plt.title('x')
    plt.subplot(1,3,2)
    plt.plot(testData_cDF['y'].iloc[::subSample],'o', color=[1,0,0,.5] )
    plt.plot(trainData_cDF['y'].iloc[::subSample],'x')
    plt.legend(['testData', 'trainData'])
    plt.title('y')
    plt.subplot(1,3,3)
    plt.plot(testData_cDF['z'].iloc[::subSample],'o', color=[1,0,0,.5] )
    plt.plot(trainData_cDF['z'].iloc[::subSample],'x')
    plt.legend(['testData', 'trainData'])
    plt.title('z')
    plt.show()    

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

