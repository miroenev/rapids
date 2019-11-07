import numpy as np; import numpy.matlib
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipyvolume as ipv

import time
import copy

import cupy
import cudf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets; from sklearn.metrics import confusion_matrix, accuracy_score

import dask
from dask import delayed
from dask.distributed import as_completed

import xgboost

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

''' ---------------------------------------------------------------------
>  HPO / PARTICLE SWARM
----------------------------------------------------------------------'''

''' -------------------------------------------------------------------------
>  HPO VISUALIZATION
------------------------------------------------------------------------- '''

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


'''
----ASYNC EDITS
'''

def print_particle_stats( particleID, particleHistory, tag = '' ):
    nUpdates = len( particleHistory[particleID]['numEvaluations'] )
    particleStats = [ np.mean( particleHistory[particleID]['particleParams'], axis = 0 ), 
                      np.std( particleHistory[particleID]['particleParams'], axis = 0 ),
                      np.mean( particleHistory[particleID]['bestIterationNTrees'], axis = 0 ),
                      np.std( particleHistory[particleID]['bestIterationNTrees'], axis = 0 )]    
    
    print( f'    particleID : {particleID} : {nUpdates} updates : {tag}   \
       \n --------------: ----------------------------  \
       \n     parameter : mean +/- standard deviation \
       \n --------------: ---------------------------- \
       \n         depth : {particleStats[0][0]:.8f} +/- {particleStats[1][0]:.2f} \
       \n learning rate : {particleStats[0][1]:.8f}  +/- {particleStats[1][1]:.2f} \
       \n         gamma : {particleStats[0][2]:.8f} +/- {particleStats[1][2]:.2f} \
       \n        nTrees : {particleStats[2]:.2f} +/- {particleStats[3]:.2f} \n')
    
def plot_eval_distribution( particleHistory, globalBestParticleID, showStats = False ) :
    sortedBarHeightsDF = sorted_eval_frequency_per_particle ( particleHistory )
    bestParticleAccuracy = np.max( particleHistory[globalBestParticleID]['particleAccuracyTest'] )
    ax = sortedBarHeightsDF.plot.bar( figsize = (15,5), 
                                      fontsize = 16, color = [tuple(rapidsColors[1])],
                                      title = f'best accuracy: {bestParticleAccuracy:.4f}, \n total evals {sortedBarHeightsDF.sum().values}')

    ax.set ( xlabel = 'particle ID', ylabel = 'nEvals');
    ax.grid(True, which = 'major', axis = 'y', alpha=.7,  linestyle='-', color = [.75, .75, .75])
    
    if showStats:
        mostUpdatedParticle = sortedBarHeightsDF['nEvals'].index[0]
        print_particle_stats ( mostUpdatedParticle, particleHistory, tag = 'most-updated' )
        leastUpdatedParticle = sortedBarHeightsDF['nEvals'].index[-1]
        print_particle_stats ( leastUpdatedParticle, particleHistory, tag = 'least-updated' )
        
    return sortedBarHeightsDF

def sorted_eval_frequency_per_particle ( particleHistory ):
    barHeights = {}
    for particleKey in particleHistory.keys():
        if isinstance( particleKey, int ):
            barHeights[particleKey] = len( particleHistory[particleKey]['numEvaluations'] )

    barHeightsDF = pd.DataFrame.from_dict( data = barHeights, columns = ['nEvals'], orient='index' )

    sortedBarHeightsDF = barHeightsDF.sort_values(by = 'nEvals', ascending=False)
    return sortedBarHeightsDF


def sample_params (paramRanges, randomSeed = None):
    #if randomSeed:
    #    np.random.seed(randomSeed)
    
    paramSamples = []
    velocitySamples = []
    
    for iParam, paramRange in paramRanges.items():        
        lowerBound = paramRange[1]
        upperBound = paramRange[2]
        paramType = paramRange[3]

        if paramType == 'int':
            paramSamples += [ np.random.randint(low = lowerBound, high = upperBound) ]
            
        elif paramType =='float':            
            paramSamples += [ np.random.uniform(low = lowerBound, high = upperBound) ]
                
        velocitySamples += [ np.random.uniform( low = -np.abs( upperBound - lowerBound ), high = np.abs( upperBound - lowerBound ) ) ]
        
    return np.array(paramSamples), np.array( velocitySamples)

def initialize_particle_swarm ( nParticles, paramRanges, randomSeed = None, plotFlag = False ):
    
    if randomSeed is not None:
        np.random.seed(randomSeed)

    particleParams = []
    particleVelocities = []
    
    for iParticle in range ( nParticles ):        
        paramSamples, velocitySamples = sample_params ( paramRanges )        
        particleVelocities += [ velocitySamples ]
        particleParams += [ paramSamples ]
        
    particleColors = np.random.rand( nParticles, 3)
    
    globalBest = {'accuracy': 0,
                  'particleID': -1,
                  'params': [],
                  'iEvaluation': - 1}    
    
    if plotFlag:
        particles = np.array( particleParams )
        velocities = np.array( particleVelocities )        
        quiverSizes = np.clip( np.linalg.norm( velocities, axis = 1 ), 5, 15)
        
        ipv.figure()
        ipv.quiver( particles[:,0], particles[:,1], particles[:,2], 
                    velocities[:,0], velocities[:,1], velocities[:,2], 
                    color = particleColors, size = quiverSizes)
        ipv.xlim( paramRanges[0][1], paramRanges[0][2] )
        ipv.ylim( paramRanges[1][1], paramRanges[1][2] )
        ipv.zlim( paramRanges[2][1], paramRanges[2][2] )
        ipv.show()
        
    return particleParams, particleVelocities, globalBest, particleColors

def initialize_particle_history ( particleID, particleParams, particleHistory ):
    particleHistory[particleID] = { 'numEvaluations': [], 
                                    'particleAccuracyTest': [], 
                                    'particleAccuracyTrain': [],
                                    'particleParams': [],
                                    'particleVelocities': [], 
                                    'bestIterationNTrees': [],
                                    'personalBestAccuracyTest': 0,
                                    'personalBestParams': np.zeros(particleParams.shape),
                                    'predictions': [] }
    return particleHistory

def format_params( particleParams, nTrees ):
    return f"{int(particleParams[0]):d}, {particleParams[1]:.4f}, {particleParams[2]:.4f}, {nTrees}"

def enforce_param_bounds_inline ( particleParams, paramRanges  ):
    for iParam in range( particleParams.shape[0] ):
        particleParams[ iParam ] = np.clip( particleParams[ iParam ], paramRanges[iParam][1], paramRanges[iParam][2])
        if paramRanges[iParam][3] == 'int':
            particleParams[ iParam ] = np.round( particleParams[ iParam ] )
    return particleParams


def evaluate_particle ( particle, dataFutures, earlyStoppingRounds, retainPredictionsFlag ):    

    # fixed parameters
    paramsGPU = { 'objective': 'binary:hinge',
                  'tree_method': 'gpu_hist',
                  'random_state': 0 }

    # TODO: loop over paramRanges instead of hard code
    paramsGPU['max_depth'] = int( particle['params'][0] )
    paramsGPU['learning_rate'] = particle['params'][1]
    paramsGPU['gamma'] = particle['params'][2]
    paramsGPU['num_boost_rounds'] = 1000

    startTime = time.time()

    trainDMatrix = xgboost.DMatrix( data = dataFutures['trainData'], label = dataFutures['trainLabels'] )
    testDMatrix = xgboost.DMatrix( data = dataFutures['testData'], label = dataFutures['testLabels'] )

    trainedModelGPU = xgboost.train( dtrain = trainDMatrix, evals = [(testDMatrix, 'test')], 
                                     params = paramsGPU,
                                     num_boost_round = paramsGPU['num_boost_rounds'],
                                     early_stopping_rounds = earlyStoppingRounds,
                                     verbose_eval = False )
        
    predictionsGPU = trainedModelGPU.predict( testDMatrix ).astype(int)
            
    elapsedTime = time.time() - startTime

    particle['nTrees'] = trainedModelGPU.best_iteration
    particle['trainAccuracy'] = 1 - float( trainedModelGPU.eval(trainDMatrix, iteration = 50).split(':')[1] )
    particle['testAccuracy'] = 1 - float( trainedModelGPU.eval(testDMatrix, iteration = 50).split(':')[1] )    
    
    if not retainPredictionsFlag: 
        predictionsGPU = None
    
    particle['predictions'] = predictionsGPU
    
    return particle, elapsedTime


def update_particle( particle, paramRanges, globalBestParams, personalBestParams, 
                     wMomentum, wIndividual, wSocial, wExplore, randomSearchMode = False, randomSeed = None ):
    ''' 
    # TODO: debug dask caching [?] attempting to use a seed produces the same sequence of random samples
    if randomSeed is not None:
        np.random.seed(randomSeed)    
    '''
    
    # baseline to compare swarm update versus random search
    if randomSearchMode:        
        sampledParams, sampledVelocities = sample_params( paramRanges )
        return sampledParams, sampledVelocities
        
    # computing update terms for particle swarm
    inertiaInfluence = particle['velocities'].copy()
    socialInfluence = ( globalBestParams - particle['params'] )
    individualInfluence = ( personalBestParams - particle['params'] )
    
    newParticleVelocities =    wMomentum    *  inertiaInfluence \
                             + wIndividual  *  individualInfluence  * np.random.random()   \
                             + wSocial      *  socialInfluence      * np.random.random()
    
    newParticleParams = particle['params'].copy() + newParticleVelocities
    newParticleParams = enforce_param_bounds_inline ( newParticleParams, paramRanges )
            
    return newParticleParams, newParticleVelocities


def run_hpo ( client, mode, paramRanges, trainData_cDF, trainLabels_cDF, testData_cDF, testLabels_cDF,
              nParticles, nEpochs,             
              wMomentum = .05, wIndividual = .35, wBest = .25, wExplore = .15, earlyStoppingRounds = 50,
              terminationAccuracy = np.Inf, 
              randomSeed = 0, 
              plotFlag = False,
              retainPredictionsFlag = False ):
    
    startTime = time.time()
    
    # ----------------------------
    # scatter data to all workers
    # ----------------------------
    if client is not None:
        scatteredData_future = client.scatter( [ trainData_cDF, trainLabels_cDF, testData_cDF, testLabels_cDF], broadcast = True )
        
    dataFutures = { 'trainData'   : scatteredData_future[0], 'trainLabels' : scatteredData_future[1], 
                    'testData'    : scatteredData_future[2], 'testLabels'  : scatteredData_future[3] }
    
    # ----------------------------
    # initialize HPO strategy 
    # ----------------------------        
    def initialize_particle_futures ( nParticles, paramRanges, randomSeed, plotFlag ) :
        initialParticleParams, initialParticleVelocities, globalBest, particleColors = initialize_particle_swarm ( nParticles, paramRanges, randomSeed, plotFlag )
        # create particle futures using the initialization positions and velocities    
        delayedEvalParticles = []
        for iParticle in range(nParticles):
            particle = { 'ID': iParticle, 'params': initialParticleParams[iParticle], 'velocities': initialParticleVelocities[iParticle], 'predictions': None }        
            delayedEvalParticles.append( delayed ( evaluate_particle )( particle.copy(), dataFutures, earlyStoppingRounds, retainPredictionsFlag ))
        return delayedEvalParticles, initialParticleParams, globalBest, particleColors
        
    # ------------------------------------------------
    # shared logic for particle evaluation and updates
    # ------------------------------------------------
    def eval_and_update ( particleFuture, delayedEvalParticles, particleHistory, paramRanges, globalBest, randomSearchMode, nEvaluations ):
        # convert particle future to concrete result and collect returned values
        particle, elapsedTime = particleFuture.result()

        # update hpo strategy meta-parameters -- i.e. swarm global best and particle personal best
        particleHistory, globalBest = update_bests ( particleHistory, particle, globalBest, nEvaluations, mode['randomSearch'] )

        # update history with this particle's latest contribution/eval
        particleHistory = update_history_dictionary ( particleHistory, particle, nEvaluations )

        # update particle
        if randomSearchMode:
            personalBestParams = None
        else:
            personalBestParams = particleHistory[particle['ID']]['personalBestParams']
            
        particle['params'], particle['velocities'] = update_particle ( particle, paramRanges,
                                                                       globalBest['params'], personalBestParams,
                                                                       wMomentum, wIndividual, wBest, wExplore,
                                                                       randomSearchMode = randomSearchMode, 
                                                                       randomSeed = particle['ID'] ) # repeatability
        return particle.copy(), particleHistory, globalBest
    
    nEvaluations = 0
    particleHistory = {}
    
    if mode['allowAsyncUpdates'] != True:
        # ----------------------------
        # synchronous particle swarm
        # ----------------------------
        delayedEvalParticles, initialParticleParams, globalBest, particleColors = initialize_particle_futures ( nParticles, paramRanges, randomSeed, plotFlag )
        futureEvalParticles = client.compute( delayedEvalParticles )
        
        for iEpoch in range (0, nEpochs ):    
            futureEvalParticles = client.compute( delayedEvalParticles )
            delayedEvalParticles = []
            for particleFuture in futureEvalParticles:
                newParticle, particleHistory, globalBest = eval_and_update ( particleFuture, delayedEvalParticles, particleHistory, paramRanges, globalBest, mode['randomSearch'], nEvaluations )

                # termination conditions 
                if globalBest['accuracy'] > terminationAccuracy: break

                # append future work for the next instantiation of this particle ( using the freshly updated parameters )
                delayedEvalParticles.append( delayed ( evaluate_particle )( newParticle, dataFutures, earlyStoppingRounds, retainPredictionsFlag ))
                
                nEvaluations += 1
            # --- 
            print(f' > on epoch {iEpoch} out of {nEpochs}') 
    
    else:
        # ----------------------------
        # asynchronous particle swarm
        # ----------------------------
        delayedEvalParticles, initialParticleParams, globalBest, particleColors = initialize_particle_futures ( nParticles, paramRanges, randomSeed, plotFlag )
        futureEvalParticles = client.compute( delayedEvalParticles )        
        particleFutureSeq = as_completed( futureEvalParticles )
        
        for particleFuture in particleFutureSeq:
            newParticle, particleHistory, globalBest = eval_and_update ( particleFuture, delayedEvalParticles, particleHistory, paramRanges, globalBest, mode['randomSearch'], nEvaluations )
            
            # termination conditions 
            if globalBest['accuracy'] > terminationAccuracy: break
            approximateEpoch = nEvaluations // nParticles
            if ( approximateEpoch ) > nEpochs : break
            
            # append future work for the next instantiation of this particle ( using the freshly updated parameters )
            delayedParticle = delayed ( evaluate_particle )( newParticle, dataFutures, earlyStoppingRounds, retainPredictionsFlag )
            # submit this particle future to the client ( returns a future )
            futureParticle = client.compute( delayedParticle )
            # track its completion via the as_completed iterator 
            particleFutureSeq.add( futureParticle )
            
            nEvaluations += 1
            if nEvaluations % nParticles == 0:
                print(f' > on approximate epoch {approximateEpoch} out of {nEpochs}') 
                              
    elapsedTime = time.time() - startTime
    
    print(f"\n\n best accuracy: {globalBest['accuracy']}, by particle: {globalBest['particleID']} on eval: {globalBest['iEvaluation']} ")
    print(f" best parameters: {format_params( globalBest['params'], globalBest['nTrees'] )}, \n elpased time: {elapsedTime:.2f} seconds")
    
    particleHistory['initialParams'] = initialParticleParams
    particleHistory['paramRanges'] = paramRanges
    particleHistory['particleColors'] = particleColors
    particleHistory['nParticles'] = nParticles
    return particleHistory, globalBest, elapsedTime


def update_bests ( particleHistory, particle, globalBest, numEvaluations = -1, randomSearchMode = False, printPersonalBestUpdates = False ):
    
    # check to see if current particle is new global best
    if particle['testAccuracy'] > globalBest['accuracy']:
        globalBest['accuracy'] = particle['testAccuracy']
        globalBest['params'] = particle['params']
        globalBest['particleID'] = particle['ID']
        globalBest['nTrees'] = particle['nTrees']
        globalBest['iEvaluation'] = numEvaluations        
        
        print(f"{globalBest['accuracy']:.6f} -- new global best by particle: {globalBest['particleID']}, eval: {numEvaluations}, params: {format_params( globalBest['params'], globalBest['nTrees'] )}")
    
    if particle['ID'] not in particleHistory.keys(): 
        particleHistory = initialize_particle_history ( particle['ID'], particle['params'], particleHistory )
            
    if randomSearchMode:
        #print('random-search-mode : skipping updates to personal best')
        pass
    else:
        # update personal best
        if particle['testAccuracy'] > particleHistory[particle['ID']]['personalBestAccuracyTest']:
            particleHistory[particle['ID']]['personalBestAccuracyTest'] = particle['testAccuracy']
            particleHistory[particle['ID']]['personalBestParams'] = particle['params']            

            if printPersonalBestUpdates:
                print ( f"   - new personal best for particle {particle['ID']}, {particleHistory[particle['ID']]['personalBestAccuracyTest']:.4f}, params: {format_params( particle['params'], particle['nTrees'] )}")
    
    return particleHistory, globalBest

def update_history_dictionary( particleHistory, particle, numEvaluations, testDataStride = 10 ):
    
    if particle['ID'] not in particleHistory.keys():
        particleHistory = initialize_particle_history ( particle['ID'], particle['params'], particleHistory )
    
    particleHistory[particle['ID']]['numEvaluations'] += [ numEvaluations ]
    
    if particle['predictions'] is not None:
        if particle['testAccuracy'] > np.max(np.array(particleHistory[particle['ID']]['particleAccuracyTest'])):
            # new best
            particleHistory[particle['ID']]['predictions'] += [ particle['predictions'] ]
    
    particleHistory[particle['ID']]['particleAccuracyTest'] += [ particle['testAccuracy'] ] 
    particleHistory[particle['ID']]['particleAccuracyTrain'] += [ particle['trainAccuracy'] ] 
    particleHistory[particle['ID']]['particleParams'] += [ particle['params'] ] 
    particleHistory[particle['ID']]['particleVelocities'] += [ particle['velocities'] ]
    particleHistory[particle['ID']]['bestIterationNTrees'] += [ particle['nTrees'] ]    
        
    return particleHistory


def viz_particle_movement( particleHistory ):
    
    sortedBarHeightsDF = sorted_eval_frequency_per_particle ( particleHistory )
    
    particleHistoryCopy = copy.deepcopy( particleHistory )
    
    paramRanges = particleHistoryCopy['paramRanges']
    particleColors = particleHistoryCopy['particleColors']
    nParticles = particleHistory['nParticles']
    initialParticleParams = particleHistoryCopy['initialParams']
    
    nAnimationFrames = max( sortedBarHeightsDF['nEvals'] )
    particleXYZ = np.zeros( ( nAnimationFrames, nParticles, 3 ) )
    lastKnownLocation = {}
    
    
    # TODO: bestIterationNTrees
    # particleSizes[ iFrame, iParticle ] = particleHistoryCopy[iParticle]['bestIterationNTrees'].pop(0).copy()
    
    for iFrame in range( nAnimationFrames ):
        for iParticle in range( nParticles ):
            if iParticle in particleHistoryCopy.keys():
                # particle exists in the particleHistory and it has parameters for the current frame
                if len( particleHistoryCopy[iParticle]['particleParams'] ):
                    particleXYZ[iFrame, iParticle, : ] = particleHistoryCopy[iParticle]['particleParams'].pop(0).copy()
                    lastKnownLocation[iParticle] = particleXYZ[iFrame, iParticle, : ].copy()
                else:
                    # particle exists but it's params have all been popped off -- use its last known location
                    particleXYZ[iFrame, iParticle, : ] = lastKnownLocation[iParticle].copy()
                    
            else:
                # particle does not exist in the particleHistory
                if iParticle in lastKnownLocation.keys():
                    # particle has no params in current frame, attempting to use last known location
                    particleXYZ[iFrame, iParticle, : ] = lastKnownLocation[iParticle].copy()                    
                else:
                    # using initial params
                    particleXYZ[iFrame, iParticle, : ] = initialParticleParams[iParticle].copy()
                    lastKnownLocation[iParticle] = particleXYZ[iFrame, iParticle, : ].copy()                    
        
    ipv.figure()
    
    colorStack = np.random.random( ( nParticles, 3) )
    scatterPlots = ipv.scatter( particleXYZ[:, :,0], 
                                particleXYZ[:, :,1], 
                                particleXYZ[:, :,2], 
                                marker='sphere', 
                                size=5,
                                color = particleColors )
    
    ipv.animation_control( [ scatterPlots ] , interval = 400 )
    ipv.xlim( paramRanges[0][1]-.5, paramRanges[0][2]+.5 )
    ipv.ylim( paramRanges[1][1]-.1, paramRanges[1][2]+.1 )
    ipv.zlim( paramRanges[2][1]-.1, paramRanges[2][2]+.1 )
    
    ipv.show()    
    

''' ---------------------------------------------------------------------
>  DYNAMIC CODE LOADING FROM NOTEBOOK
----------------------------------------------------------------------'''
import inspect
def import_library_function_in_new_cell ( functionPointerList, warningFlag = True ):
    
    s = ''
    if warningFlag:
        s = '# note: code below is imported from library\n'
        s += '#! if you plan to make changes be sure to fix any references to point to this local version\n'

    for iFunctionPointer in functionPointerList:
        print( iFunctionPointer )
        try: 
            s += '\n' + inspect.getsource( iFunctionPointer ) + '\n'
        except sourceLoadError:
            s += '\n unable to load ' + str(iFunctionPointer) + '\n'            

    get_ipython().set_next_input(s)
        
    return

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

def query_log ( log, queryKey, verbose = True ):
    values = [value for key, value in log.items() if re.search('^'+queryKey, key) ]
    if verbose:
        print(queryKey, '\n', pd.Series(values).describe(), '\n',)    
        plt.plot(values, 'x-')
        plt.title(str(queryKey))
    return values


