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
        barHeights[particleKey] = len( particleHistory[particleKey]['numEvaluations'] )

    

    barHeightsDF = pd.DataFrame.from_dict( data = barHeights, columns = ['nEvals'], orient='index' )

    sortedBarHeightsDF = barHeightsDF.sort_values(by = 'nEvals', ascending=False)
    return sortedBarHeightsDF


def sample_params (paramRanges, randomSeed = None):
    
    if randomSeed:
        np.random.seed(randomSeed)
    
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

def initialize_particle_swarm ( nParticles, paramRanges, randomSeed = None, plotFlag = True ):
    
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
        ipv.quiver( particles[:,0], particles[:,1], particles[:,2], velocities[:,0], velocities[:,1], velocities[:,2], color = particleColors, size = quiverSizes)
        ipv.xlim( paramRanges[0][1], paramRanges[0][2] );  ipv.ylim( paramRanges[1][1], paramRanges[1][2] );  ipv.zlim( paramRanges[2][1], paramRanges[2][2] )
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
                                    'particleTestDataPredictions': [] }
    return particleHistory

def format_params( particleParams, nTrees ):
    return f"{int(particleParams[0]):d}, {particleParams[1]:.4f}, {particleParams[2]:.4f}, {nTrees}"

def enforce_param_bounds_inline ( particleParams, paramRanges  ):
    for iParam in range( particleParams.shape[0] ):
        particleParams[ iParam ] = np.clip( particleParams[ iParam ], paramRanges[iParam][1], paramRanges[iParam][2])
        if paramRanges[iParam][3] == 'int':
            particleParams[ iParam ] = np.round( particleParams[ iParam ] )
    return particleParams

def update_bests ( particleHistory, particle, globalBest, numEvaluations = -1, randomSearchMode = False, printPersonalBestUpdates = True ):
    
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
        print('random-search-mode : skipping updates to personal best')
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
    
    if len( particleHistory[particle['ID']]['particleAccuracyTest'] ) and particle['predictions'] is not None:
        if particle['testAccuracy'] > np.max(np.array(particleHistory[particle['ID']]['particleAccuracyTest'])):
            # new best
            particleHistory[particle['ID']]['particleTestDataPredictions'] += [ particle['predictions'] ]
    
    particleHistory[particle['ID']]['particleAccuracyTest'] += [ particle['testAccuracy'] ] 
    particleHistory[particle['ID']]['particleAccuracyTrain'] += [ particle['trainAccuracy'] ] 
    particleHistory[particle['ID']]['particleParams'] += [ particle['params'] ] 
    particleHistory[particle['ID']]['particleVelocities'] += [ particle['velocities'] ]
    particleHistory[particle['ID']]['bestIterationNTrees'] += [ particle['nTrees'] ]    
        
    return particleHistory


def viz_particle_movement( particleHistory, colorStack = [] ):
    
    sortedBarHeightsDF = sorted_eval_frequency_per_particle ( particleHistory )
    nParticles = len(particleHistory)
    particleHistoryCopy = copy.deepcopy( particleHistory )

    nAnimationFrames = max( sortedBarHeightsDF['nEvals'] )

    particleXYZ = np.zeros( ( nAnimationFrames, nParticles, 3 ) )
    lastKnownLocation = {}

    for iFrame in range( nAnimationFrames ):
        for iParticle in range( nParticles ):
            if iParticle in particleHistoryCopy.keys():
                if len( particleHistoryCopy[iParticle]['particleParams'] ):
                    particleXYZ[iFrame, iParticle, : ] = particleHistoryCopy[iParticle]['particleParams'].pop(0).copy()
                    lastKnownLocation[iParticle] = particleXYZ[iFrame, iParticle, : ].copy()
                else:
                    if iParticle in lastKnownLocation.keys():
                        particleXYZ[iFrame, iParticle, : ] = lastKnownLocation[iParticle].copy()

    # TODO: trajectory plot

    ipv.figure()

    colorStack = np.random.random( ( nParticles, 3) )
    scatterPlots = ipv.scatter( particleXYZ[:, :,0], particleXYZ[:, :,1], particleXYZ[:, :,2], marker='sphere', size=5, color = colorStack )

    ipv.animation_control( [scatterPlots ] )
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


