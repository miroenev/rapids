import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
import ipyvolume as ipv
import numpy as np
import cudf
import time
import copy

# plotting params
ipvPlotWidth = 800
ipvPlotHeight = 600

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
                 5: rapidsSecondary5,
                 6: np.hstack( [ np.random.random(3), np.array(1) ] ),
                 7: np.hstack( [ np.random.random(3), np.array(1) ] ),
                 8: np.hstack( [ np.random.random(3), np.array(1) ] ),
                 9: np.hstack( [ np.random.random(3), np.array(1) ] ) }

def reduce_to_3D ( data, labels, datasetName, 
                   predictedLabels = None, 
                   dimReductionMethod = 'UMAP', 
                   maxSamplesForDimReduction = None ):
    
    startTime = time.time()        
    
    # decimate to limit dim-reduction out of memory errors
    if maxSamplesForDimReduction is None:
        decimationFactor = 1        
    else:        
        decimationFactor = np.max((1, data.shape[0] // maxSamplesForDimReduction))
    
    if datasetName == 'synthetic': # TODO: assumes non-shuffled data, test with shuffled
        decimationFactor *= 2
        coilData_1 = data.iloc[0::decimationFactor]
        coilData_2 = data.iloc[1::decimationFactor]
        coilLabel_1 = labels.iloc[0::decimationFactor]
        coilLabel_2 = labels.iloc[1::decimationFactor]
        decimatedData = cudf.concat( [ coilData_1, coilData_2] )
        decimatedLabels = cudf.concat( [ coilLabel_1, coilLabel_2] )
    else:
        decimatedData = data[::decimationFactor]
        decimatedLabels = labels[::decimationFactor]
        
    print(f' > dim. reduction using {dimReductionMethod}', end = ' -- ')
    if decimationFactor > 1:
        print(f'decimating by {decimationFactor}x', end = ' -- ' )
        
    if dimReductionMethod == 'PCA':        
        dimReductionModel = cuml.PCA( copy = True, n_components = 3,
                                      random_state = 0, 
                                      svd_solver = 'full', 
                                      verbose = True, 
                                      whiten = False )
        embeddedData = dimReductionModel.fit_transform ( X = decimatedData )

    elif dimReductionMethod == 'TSNE':        
        embeddedData = cuml.TSNE( n_components = 2 ).fit_transform ( X = decimatedData )
        embeddedData.add_column('z', cudf.Series(np.zeros((decimatedData.shape[0]))) )
        
    elif dimReductionMethod == 'UMAP':        
        embeddedData = cuml.UMAP( n_components = 3 ).fit_transform( X = decimatedData, y = decimatedLabels )
                          
    # decimate predicted labels to match embedded data + labels
    if predictedLabels is not None:
        decimatedPredictions = predictedLabels[::decimationFactor] 
    else:
        decimatedPredictions = None

    elapsedTime = time.time() - startTime
    print(f'new shape: {embeddedData.shape} -- completed in: {elapsedTime:.3f} seconds')

    return embeddedData, decimatedLabels, decimatedPredictions


def plot_data( data, labels, datasetName, predictedLabels = None,
               dimReductionMethod = 'UMAP', 
               maxSamplesForDimReduction = 1000000, 
               maxSamplesToPlot = 100000 ):
    '''
    assumes binary labels
    '''
    
    print(f'plotting {datasetName.upper()} dataset, original shape: {data.shape}')
    
    if data.shape[1] != 3:
        embeddedData, decimatedLabels, decimatedPredictions = reduce_to_3D ( data, labels, datasetName,
                                                                            predictedLabels, dimReductionMethod,
                                                                            maxSamplesForDimReduction )
    else:
        embeddedData = data
        decimatedLabels = labels
        decimatedPredictions = predictedLabels
    
    # decimation for visualizaiton
    maxSamplesToPlot = np.min( ( maxSamplesToPlot, embeddedData.shape[0] ))
        
    if embeddedData.shape[0] != maxSamplesToPlot:
        percentOfTotal = maxSamplesToPlot/embeddedData.shape[0] * 100.
        print( f' > plotting subset of {maxSamplesToPlot} samples -- {percentOfTotal:0.2f}% of total, adjust via maxSamplesToPlot ')
        
        # read a maxSamplesToPlot number of samples (shuffled) and convert to numpy [CPU] for plotting
        targetSamples = np.random.permutation(embeddedData.shape[0])[0:maxSamplesToPlot]
        xyzViz = embeddedData.iloc[targetSamples, :].as_matrix()  # note implicit copy in the cudf->numpy conversion
        labelsViz = decimatedLabels.iloc[targetSamples].copy()
        if decimatedPredictions is not None:
            predictionsViz = decimatedPredictions.iloc[targetSamples]
        else:
            predictionsViz = None
    else:
        xyzViz = embeddedData.as_matrix()
        labelsViz = decimatedLabels
        predictionsViz = decimatedPredictions
    
    # apply colors   
    nClasses = labelsViz[labelsViz.columns[0]].nunique()
    
    for iClass in range ( nClasses ):
        boolMask = labelsViz == iClass
        boolMask = boolMask.as_matrix()
        
        if iClass == 0:
            color = boolMask * np.ones ( (xyzViz.shape[0], 4) )  * rapidsColors[iClass]
        else:
            color += boolMask * np.ones ( (xyzViz.shape[0], 4) )  * rapidsColors[iClass]
    
    ipv.figure ( width = 800, height = 600 )
    ipv.scatter ( xyzViz[:,0], xyzViz[:,1], xyzViz[:,2], color = color, marker = 'sphere', size = .5)
    
    if predictionsViz is not None:
        npLabels = np.squeeze(labelsViz.as_matrix().astype(int))
        mispredictedSamples = np.where ( npLabels != predictionsViz )[0]        
        ipv.scatter( xyzViz[mispredictedSamples,0], 
                     xyzViz[mispredictedSamples,1], 
                     xyzViz[mispredictedSamples,2], color = [1, 1, 1, .9], marker = 'diamond', size = .5)
    
    ipv.pylab.squarelim()
    ipv.show()

''' -------------------------------------------------------------------------
>  post swarm viz
------------------------------------------------------------------------- '''
    
def viz_prep( swarm ):
    nEvalList = []
    evalTimeList = []
    pIDList = []
    particleHistory = {}
    for pID, particle in swarm.particles.items():
        nEvalList.append ( particle.nEvals )
        evalTimeList.append ( np.mean( particle.evalTimeHistory ) )
        particleHistory[pID] = particle.posHistory
        pIDList.append( pID )
        
    swarmDF = pd.DataFrame ( data = { 'nEvals' : nEvalList}, index = pIDList ).sort_values('nEvals', ascending=False)
    
    return swarmDF, particleHistory    



#def assign_particle_colors( swarm, colormap, topN = None ):
    

def viz_particle_trails( swarm, topN = None, globalBestIndicatorFlag = True ):

    startTime = time.time()
    
    minParticleSize = .75; maxParticleSize = 5; nTreesDescaler = 100
    
    ipv.figure( width = 800, height = 600 )
    
    # find top performing particles to show additional detail [ size scaled based on nTrees ] 
    if topN is None: topN = swarm.nParticles
    topN = np.max((1, topN))
    particleMaxes = {}
    for key, iParticle in swarm.particles.items():
        if len(iParticle.testDataPerfHistory):
            particleMaxes[key] = max(iParticle.testDataPerfHistory)
        else:
            particleMaxes[key] = 0
            
    sortedParticles = pd.DataFrame.from_dict(particleMaxes, orient='index').sort_values(by=0,ascending=False)
    topParticles = sortedParticles.index[0:np.min((np.max((topN,1)),swarm.nParticles))]  
        
    topParticleCounter = 0
    for iParticle in range( swarm.nParticles ):

        if len( swarm.particles[ iParticle ].posHistory ):
            particlePosHistory = np.matrix( swarm.particles[ iParticle ].posHistory )
            
            # plot markers along the particle's journey over the epoch sequence
            if iParticle in topParticles:
                # draw top particles with their unique color and scaled by the number of trees
                particleColor = swarm.particles[iParticle].color # colorSet[topParticleCounter]
                topParticleCounter += 1
                colorAlpha = 1
                for iEval in range( swarm.particles[iParticle].nEvals ):
                    particleSize = np.clip( swarm.particles[iParticle].nTreesHistory[iEval]/nTreesDescaler, minParticleSize, maxParticleSize)
                    xyz = np.matrix( particlePosHistory[iEval,:] )
                    ipv.scatter( xyz[:,0], xyz[:,1], xyz[:,2], size = particleSize,
                                 marker = 'sphere', color = particleColor, grow_limits = False )
            else:
                # draw non-top particle locations in gray
                particleColor = ( .8, .8, .8, .9 )
                xyz = np.matrix( particlePosHistory )
                ipv.scatter( xyz[:,0].squeeze(), xyz[:,1].squeeze(), xyz[:,2].squeeze(), size = .75,
                             marker = 'sphere', color = particleColor, grow_limits = False )
                
            # plot line trajectory [ applies to both top and non-top particles ]
            ipv.plot( particlePosHistory[:,0].squeeze(),
                      particlePosHistory[:,1].squeeze(),
                      particlePosHistory[:,2].squeeze(),  color = particleColor )

    # draw an intersecting set of lines/volumes to indicate the location of the best particle / parameter-set
    if globalBestIndicatorFlag:
        bestXYZ = np.matrix( swarm.globalBest['params'] )
        ipv.scatter( bestXYZ[:,0], bestXYZ[:,1], bestXYZ[:,2], 
                     color = rapidsColors[1], 
                     size = maxParticleSize//2, marker='box') 

        data = np.ones((100,4,4))
        xSpan = np.clip( (swarm.paramRanges[0][2] - swarm.paramRanges[0][1])/200, .007, 1 )
        ySpan = np.clip( (swarm.paramRanges[1][2] - swarm.paramRanges[1][1])/200, .007, 1 )
        zSpan = np.clip( (swarm.paramRanges[2][2] - swarm.paramRanges[2][1])/200, .007, 1 )
        xMin = swarm.globalBest['params'][0] - xSpan
        xMax = swarm.globalBest['params'][0] + xSpan
        yMin = swarm.globalBest['params'][1] - ySpan
        yMax = swarm.globalBest['params'][1] + ySpan
        zMin = swarm.globalBest['params'][2] - zSpan
        zMax = swarm.globalBest['params'][2] + zSpan

        ipv.volshow(data=data.T, opacity=.15, level=[0.25,0.,0.25], extent=[[xMin,xMax],[yMin,yMax],[swarm.paramRanges[2][1],swarm.paramRanges[2][2]]], controls=False)
        ipv.volshow(data=data.T, opacity=.15, level=[0.25,0.,0.25], extent=[[swarm.paramRanges[0][1],swarm.paramRanges[0][2]],[yMin,yMax],[zMin,zMax]], controls=False)

    print(f'elapsed time {time.time()-startTime:0.2f}')
    ipv.show()

def plot_particle_evals ( swarm ):
    swarmDF, particleHistory = viz_prep( swarm )
    swarmDF.plot.bar( figsize = (15, 7), 
                      color=[ tuple(rapidsColors[1]) ],
                      edgecolor= [ tuple(rapidsColors[4]) ], zorder=3);
    plt.xlabel('pID'); plt.ylabel('nEvals');
    plt.grid(True, zorder=0)
    plt.title('Sync Swarm Evaluations Per Particle');    
    
def viz_swarm( swarm, paramRanges ):    
    swarmDF, particleHistory = viz_prep(swarm)
    
    particleHistoryCopy = copy.deepcopy( particleHistory )    
    nParticles = swarm.nParticles
    
    nAnimationFrames = list(swarmDF['nEvals'])[0] # max( sortedBarHeightsDF['nEvals'] )
    particleXYZ = np.zeros( ( nAnimationFrames, nParticles, 3 ) )
    lastKnownLocation = {}
    
    # TODO: bestIterationNTrees
    # particleSizes[ iFrame, iParticle ] = particleHistoryCopy[iParticle]['bestIterationNTrees'].pop(0).copy()
    
    for iFrame in range( nAnimationFrames ):
        for iParticle in range( nParticles ):
            if iParticle in particleHistoryCopy.keys():
                # particle exists in the particleHistory and it has parameters for the current frame
                if len( particleHistoryCopy[iParticle] ):
                    particleXYZ[iFrame, iParticle, : ] = particleHistoryCopy[iParticle].pop(0).copy()
                    lastKnownLocation[iParticle] = particleXYZ[iFrame, iParticle, : ].copy()
                else:
                    # particle exists but it's params have all been popped off -- use its last known location
                    if iParticle in lastKnownLocation.keys():
                        particleXYZ[iFrame, iParticle, : ] = lastKnownLocation[iParticle].copy()
                    
            else:
                # particle does not exist in the particleHistory
                if iParticle in lastKnownLocation.keys():
                    # particle has no params in current frame, attempting to use last known location
                    particleXYZ[iFrame, iParticle, : ] = lastKnownLocation[iParticle].copy()                    
                else:
                    print('particle never ran should we even display it')
                    assert(False)
                    # using initial params
                    #particleXYZ[iFrame, iParticle, : ] = initialParticleParams[iParticle].copy()
                    #lastKnownLocation[iParticle] = particleXYZ[iFrame, iParticle, : ].copy()                    
        
    ipv.figure()
    
    scatterPlots = ipv.scatter( particleXYZ[:, :, 0], 
                                particleXYZ[:, :, 1], 
                                particleXYZ[:, :, 2], 
                                marker = 'sphere', size = 5,
                                color = swarm.particleColorStack )
    
    ipv.animation_control( [ scatterPlots ] , interval = 400 )
    ipv.xlim( paramRanges[0][1]-.5, paramRanges[0][2]+.5 )
    ipv.ylim( paramRanges[1][1]-.1, paramRanges[1][2]+.1 )
    ipv.zlim( paramRanges[2][1]-.1, paramRanges[2][2]+.1 )
    
    ipv.show()
    
''' -------------------------------------------------------------------------
>  visualize impact of various parameters on synthetic data
------------------------------------------------------------------------- '''
def visualize_synthetic_data_variants ( coilType = 'helix', nCoils = [3, 6, 9], nSamples = 10000, sdevScales = [ .1, .1, .1], decimation = 2):
    
    decimation = np.clip( decimation, 2, 20 )
    
    fig = plt.figure( figsize=(10+5*len(nCoils), 9 ))
    plt.subplots_adjust( left=0, bottom=0, right=.95, top=.95, wspace=.01, hspace=.01 )
    
    iSubplot = 1
    for iCoils in nCoils:
        x, y, z, c = gen_synthetic_dataset_variant( coilType, iCoils, nSamples, sdevScales, decimation )
        ax3D = fig.add_subplot(1, len(nCoils), iSubplot, projection='3d')
        ax3D.scatter( x, y, z, c=c) 
        ax3D.view_init( elev = 50, azim = 25 )
        ax3D.set_title( f'{coilType}, nCoils : {iCoils}', size=20)        
        
        ax3D.xaxis.pane.set_facecolor('w')
        ax3D.yaxis.pane.set_facecolor('w')
        ax3D.zaxis.pane.set_facecolor('w')
        #ax3D.set_facecolor('white')
        iSubplot += 1

def gen_synthetic_dataset_variant (coilType = 'helix', coils=9, nSamples = 100000, sdevScales = [ .1, .1, .1], decimation = 10):
    import data_utils
    data, labels, _ = data_utils.generate_dataset ( coilType = coilType, nSamples = nSamples,  coilDensity = coils,
                                                    sdevScales = sdevScales, noiseScale = 1/10., shuffleFlag = False, rSeed = 0 )
    
    # assumes unsuffled data
    colors = np.ones((data.shape[0],4)) * rapidsColors[0]    
    colors[::2] = rapidsColors[1]
    
    cpuData = data.as_matrix()
    
    # interweaving is necessary to enable correct plotting order
    # otherwise latter points [ second coil ] always overlplots on top of earlier points
    
    # interweave decimated coils [ x dimension ]
    interwovenData = np.empty((data.shape[0]//decimation * 2,))
    interwovenData[0::2] = cpuData[::decimation, 0]
    interwovenData[1::2] = cpuData[1::decimation, 0]
    x = interwovenData.copy()
    
    # interweave decimated coils [ x dimension ]
    interwovenData[0::2] = cpuData[::decimation,1]
    interwovenData[1::2] = cpuData[1::decimation,1]
    y = interwovenData.copy()
    
    # interweave decimated coils [ x dimension ]
    interwovenData[0::2] = cpuData[::decimation,2]
    interwovenData[1::2] = cpuData[1::decimation,2]
    z = interwovenData
    
    # interweave decimated coils [ x dimension ]
    c = np.empty((colors.shape[0]//decimation * 2,4))
    c[0::2] = colors[::decimation]
    c[1::2] = colors[1::decimation]
    colors = c

    return x, y, z, colors


''' -------------------------------------------------------------------------
>  VISUALIZE TRAIN & TEST + OVERLAP
------------------------------------------------------------------------- '''
from ipywidgets import FloatSlider, VBox, jslink

def plot_train_vs_test ( dataset, maxSamplesToPlot = 50000 ):
    ipv.figure( width = ipvPlotWidth, height = ipvPlotHeight )
    
    50000
    trainDecimationFactor = np.max( ( 1, dataset.trainData.shape[0] // maxSamplesToPlot) ) * 2
    testDecimationFactor = np.max( ( 1, dataset.testData.shape[0] // maxSamplesToPlot) ) * 2
    
    print( f'visualizing using train decimation factor: {trainDecimationFactor}, test decimation factor: {testDecimationFactor}')
    xyz = dataset.trainData[dataset.trainLabels.labels.eq(1)].as_matrix()
    trainCoil1Scatter = ipv.scatter ( xyz[::trainDecimationFactor,0], xyz[::trainDecimationFactor,1], xyz[::trainDecimationFactor,2],
                                      color=rapidsColors[1], marker = 'sphere', size = 1,
                                      selected=np.arange(xyz.shape[0]), 
                                      size_selected=1, color_selected=rapidsColors[1] )

    xyz = dataset.trainData[dataset.trainLabels.labels.eq(0)].as_matrix()
    trainCoil2Scatter =ipv.scatter( xyz[::trainDecimationFactor,0], xyz[::trainDecimationFactor,1], xyz[::trainDecimationFactor,2], 
                                    color=rapidsColors[0], marker = 'sphere', size = 1,
                                    selected=np.arange(xyz.shape[0]), 
                                    size_selected=1, color_selected=rapidsColors[0] ) 

    xyz = dataset.testData.as_matrix()
    testDataScatter = ipv.scatter( xyz[::testDecimationFactor,0], xyz[::testDecimationFactor,1], xyz[::testDecimationFactor,2],
                                   color='white', marker = 'sphere', size = .1, 
                                   selected=np.arange(xyz.shape[0]), 
                                   size_selected=.1, color_selected='white')

    testSizeSelected = FloatSlider(min=0.1, max=5., step=0.1, description='test size:')
    jslink((testDataScatter, 'size_selected'), (testSizeSelected, 'value'))

    trainSizeSelected = FloatSlider(min=0.1, max=5., step=0.1, description='train size:')
    jslink((trainCoil1Scatter, 'size_selected'), (trainSizeSelected, 'value'))
    jslink((trainCoil2Scatter, 'size_selected'), (trainSizeSelected, 'value'))

    if dataset.datasetName == 'synthetic':
        if dataset.coilType == 'helix':
            ipv.xlim( ipv.gcf().xlim[0], ipv.gcf().xlim[1])
            ipv.ylim( ipv.gcf().ylim[0]-5, ipv.gcf().ylim[1]+5)
            ipv.zlim( ipv.gcf().zlim[0]-5, ipv.gcf().zlim[1]+5)
    
    display( VBox([ipv.gcc(), testSizeSelected, trainSizeSelected]) )
    
    