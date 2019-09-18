import numpy as np; import numpy.matlib
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipyvolume as ipv
import time
import cupy
import cudf

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
    ipv.scatter(coil1TestData[:,0], coil1TestData[:,1], coil1TestData[:,2], size = .1, color='lightgray', marker='sphere')
    ipv.scatter(coil2TestData[:,0], coil2TestData[:,1], coil2TestData[:,2], size = .1, color='lightgray', marker='sphere')

    # test data callout
    offset = np.max((coil1TrainData[:,2])) * 3
    ipv.scatter(coil1TestData[:,0], coil1TestData[:,1], coil1TestData[:,2] + offset, size = .25, color='lightgreen', marker='sphere')
    ipv.scatter(coil2TestData[:,0], coil2TestData[:,1], coil2TestData[:,2] + offset, size = .25, color='purple', marker='sphere')
    
    ipv.pylab.squarelim()
    ipv.show()
    
    
def gen_blob_coils ( nBlobPoints = 100, 
                     nCoordinates = 100, 
                     coilType='whirl', 
                     coilDensity = 6.25, 
                     sdevScales = [ .05, .05, .05], 
                     noiseScale=1/5., shuffleFlag = False, rSeed = 0, plotFlag = True,
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

''' ---------------------------------------------------------------------
>  HPO / PARTICLE SWARM
----------------------------------------------------------------------'''
def update_velocity ( best, particlesInTimestep, velocitiesInTimestep, sBest = .5, sExplore = .5 , deltaTime = 1 ):

    nParticles = particlesInTimestep.shape[ 0 ]
    nDimensions = particlesInTimestep.shape[ 1 ]
    
    bestRep = numpy.matlib.repmat( np.array( best ).reshape( -1, 1 ), nParticles, 1).reshape( nParticles, nDimensions )
    
    velocitiesInTimestep = .25 * velocitiesInTimestep + sBest * ( bestRep - particlesInTimestep ) + sExplore * ( np.random.randn( nParticles, nDimensions ) )
    
    particlesInTimestep += velocitiesInTimestep * deltaTime 
    
    return best, particlesInTimestep, velocitiesInTimestep

def update_positions ( best, particles, beta = .5 ):
    
    nParticles = particles.shape[ 0 ]
    bestRep = numpy.matlib.repmat( np.array( best ).reshape( -1, 1 ), nParticles, 1).reshape( nParticles, 3 )
    
    #import pdb; pdb.set_trace()
    particles += ( bestRep - particles ) * ( 1 - beta ) + beta * np.random.randn( nParticles, 3 )
    
    return best, particles

def ipv_animate (particles, velocities, particleColors, particleSizes, paramRanges):
    
    nTimesteps = particles.shape[0]
    nParticles = particles.shape[1]
    
    # determine quiver sizes and colors
    veloQuiverSizes = np.zeros((nTimesteps, nParticles, 1))    
    for iTimestep in range(nTimesteps):
        veloQuiverSizes[iTimestep, :, 0 ] = np.linalg.norm(velocities[iTimestep,:,:], axis=1)    
    veloQuiverSizes = np.clip(veloQuiverSizes, 0, 2)        
    quiverColors = np.ones((nTimesteps, nParticles, 4))*.8 #veloQuiverSizes/3
    
    ipv.figure()
    #bPlots = ipv.scatter( best[:, :, 0],  best[:, :, 1],  best[:, :, 2], marker = 'sphere', size=2, color = 'red' )
    pPlots = ipv.scatter( particles[:, :, 0], particles[:, :, 1], particles[:, :, 2], marker = 'sphere', size = particleSizes, color = particleColors)
    #qvPlots = ipv.quiver( particles[:, :, 0], particles[:, :, 1], particles[:, :, 2], velocities[:, :, 0], velocities[:, :, 1], velocities[:, :, 2], size=veloQuiverSizes[:,:,0]*3, color=quiverColors)
    
    ipv.animation_control( [ pPlots ] , interval=600)
    ipv.xlim( paramRanges[0][1], paramRanges[0][2] )
    ipv.ylim( paramRanges[1][1], paramRanges[1][2] )
    ipv.zlim( paramRanges[2][1], paramRanges[2][2] )

    ipv.show()
    
def particle_search_demo (nParticles=25, nTimesteps=50, fEval='', nParameters = 3 ):
    upperBound = 5

    best = np.zeros( (nTimesteps, 1, nParameters) )
    particles = np.zeros( (nTimesteps, nParticles, nParameters ) )
    velocities = np.zeros( (nTimesteps, nParticles, nParameters ) )

    best[0, 0, : ] = np.array([.1, .2, .3])

    particles[0, :, :] = np.random.randn(nParticles, nParameters) * upperBound
    velocities[0, :, :] = np.random.randn(nParticles, nParameters) * .2
    particleColors = np.random.uniform(size=(1, nParticles, 3))
    
    for iTimestep in range( 0, nTimesteps - 1  ):   
        # random jumps
        if iTimestep % 10 == 0:
            best[iTimestep, 0, : ] = particles[iTimestep, np.random.randint(nParticles), :]  + np.random.randn(1,3) * 2

        # train on whole dataset with current particle parameters 
        # eval validation accuracy
        best[iTimestep+1, 0, : ], particles[iTimestep+1, :, :], velocities[iTimestep+1, :, :] = update_velocity( best[iTimestep, 0, : ], 
                                                                                                                 particles[iTimestep, :, :], 
                                                                                                                 velocities[iTimestep, :, :],
                                                                                                                 sBest = .5, 
                                                                                                                 sExplore = .75, 
                                                                                                                 deltaTime = 1)
        # capture history
    ipv_animate(best, particles, velocities, particleColors)
    return
    
def query_log ( log, queryKey, verbose = False ):
    values = [value for key, value in log.items() if re.search('^'+queryKey, key) ]
    if verbose:
        print(queryKey, '\n', pd.Series(values).describe(), '\n',)    
        plt.plot(values, 'x-')
    return values

def initalize_hpo ( nTimesteps, nParticles, nWorkers, paramRanges, nParameters = 3, plotFlag = True):
    accuracies = np.zeros( ( nTimesteps, nParticles) )
    bestParticleIndex = {}

    bestParticle = np.zeros( (nTimesteps, 1, nParameters) )
    particles = np.zeros( (nTimesteps, nParticles, nParameters ) )
    velocities = np.zeros( (nTimesteps, nParticles, nParameters ) )
    
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
        
    return particles, velocities, accuracies, bestParticleIndex, particleColors

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