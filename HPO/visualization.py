import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ipywidgets import FloatSlider, ColorPicker, VBox, HBox, Label, jslink
import ipywidgets as widgets

import pandas as pd
import ipyvolume as ipv
import numpy as np
import cudf
import time
import copy

# plotting params
ipvPlotWidth = 800
ipvPlotHeight = 600
figsizeWidth = 15
figsizeHeight = 7
''' -------------------------------------------------------------------------
>  define colors -- https://rapids.ai/branding.html
------------------------------------------------------------------------- '''
rapidsColorsHex = { 0: '#7400ff', # rapidsPrimary1
                    1: '#64E100', # nvidia green
                    2: '#d216d2', # rapidsPrimary2
                    3: '#36c9dd', # rapidsPrimary3
                    4: '#52ffb8',
                    5: '#335d7d',
                    6: '#bfb476',
                    7: '#bababa',
                    8: '#2e6700',
                    9: '#000000' }

def append_colormap( colorDict, cmap = matplotlib.cm.jet ):
    baseColors = len(colorDict)
    for iColor in range( baseColors, baseColors + cmap.N):    
        colorDict.update( { iColor : matplotlib.colors.rgb2hex( cmap( iColor - baseColors )) } )
    return colorDict

rapidsColorsHex = append_colormap( rapidsColorsHex )
nRapidsColors = len ( rapidsColorsHex ) 


''' -------------------------------------------------------------------------
>  dataset plotting and dimensionality reduction
>  train and test set visualization with interactive sliders/buttons
------------------------------------------------------------------------- '''
def reduce_to_3D ( data, labels, predictedLabels = None, 
                   dimReductionMethod = 'UMAP', 
                   maxSamplesForDimReduction = None ):
    
    startTime = time.time()        
    
    # decimate to limit dim-reduction out of memory errors
    if maxSamplesForDimReduction is None:
        decimationFactor = 1        
    else:        
        decimationFactor = np.max((1, data.shape[0] // maxSamplesForDimReduction))
    
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

def plot_data( dataset, plotMode = 'full', maxSamplesToPlot = 100000, 
               dimReductionMethod = 'UMAP', 
               maxSamplesForDimReduction = None, predictedLabels = None ):
    
    ipvFig = comboBox = ipv.figure ( width = ipvPlotWidth, height = ipvPlotHeight, controls=False )    
    
    if plotMode == 'full':
        scatterPlot, _ = plot_decimated_data_layer( dataset.data, dataset.labels, maxSamplesToPlot )
        
    elif plotMode == 'train':
        scatterPlot, _ = plot_decimated_data_layer( dataset.trainData, dataset.trainLabels, maxSamplesToPlot )
        
    elif plotMode == 'test':
        scatterPlot, _ = plot_decimated_data_layer( dataset.testData, dataset.testLabels, maxSamplesToPlot )
        
    elif plotMode == 'test-vs-train':
        # determine split of maxSamplesToPlot per data-subset
        trainDataFraction = dataset.testData.shape[0]/dataset.data.shape[0]
        testDataFraction = 1. - trainDataFraction
        testDataSamplesToPlot = int( maxSamplesToPlot * testDataFraction )
        trainDataSamplesToPlot = int( maxSamplesToPlot * testDataFraction )
        testDataScatterPlot, testColorStack = plot_decimated_data_layer( dataset.testData, dataset.testLabels, maxSamplesToPlot = trainDataSamplesToPlot )
        trainDataScatterPlot, trainColorStack = plot_decimated_data_layer( dataset.trainData, dataset.trainLabels,  maxSamplesToPlot = trainDataSamplesToPlot )
        
        # sliders to impact marker size of train and test data
        testSize = FloatSlider( min=0.1, max=5., step=0.1, readout_format='.1f')
        jslink((testDataScatterPlot, 'size_selected'), (testSize, 'value'))

        trainSize = FloatSlider(min=0.1, max=5., step=0.1, readout_format='.1f')
        jslink((trainDataScatterPlot, 'size_selected'), (trainSize, 'value'))
        
        # buttons to allow coloring test and train data white and reverting back
        markTestButton = widgets.Button( description = "mark TEST"); 
        unmarkTestButton = widgets.Button( description = "unmark TEST")
        markTrainButton = widgets.Button( description = "mark TRAIN"); 
        unmarkTrainButton = widgets.Button( description = "unmark TRAIN")

        def apply_test_color(_):
            testDataScatterPlot.set_trait('color_selected', 'white' )
        def revert_test_color(_):
            testDataScatterPlot.set_trait('color_selected', testColorStack )
        def apply_train_color(_):
            trainDataScatterPlot.set_trait('color_selected', 'white' )
        def revert_train_color(_):
            trainDataScatterPlot.set_trait('color_selected', trainColorStack )
        
        markTestButton.on_click( apply_test_color )
        unmarkTestButton.on_click( revert_test_color )
        markTrainButton.on_click( apply_train_color )
        unmarkTrainButton.on_click( revert_train_color )
                
        htmlString = f"<font size=5>Dataset : <font color='{rapidsColorsHex[0]}'>{dataset.datasetName.capitalize()}</font></font>"
        comboBox = VBox( [ ipvFig, widgets.HTML(value=htmlString), 
                          HBox( [ Label('TEST  marker size:'), testSize]), 
                          HBox( [ Label('TRAIN marker size:'), trainSize]), 
                          HBox( [ markTestButton, unmarkTestButton ] ), 
                          HBox( [ markTrainButton, unmarkTrainButton ] ) ]) 
        
    comboBox.layout.align_items = 'center'
    display(comboBox)

def plot_decimated_data_layer ( data, labels, maxSamplesToPlot = 100000, predictedLabels = None, noPlotFlag = False):
        
    # dimensionality reduction [ and optional decimation ]
    if data.shape[1] != 3:
        maxSamplesForDimReduction = maxSamplesToPlot
        embeddedData, decimatedLabels, decimatedPredictions = reduce_to_3D ( data, labels, datasetName,
                                                                            predictedLabels, dimReductionMethod,
                                                                            maxSamplesForDimReduction )
    else:
        embeddedData = data
        decimatedLabels = labels
        decimatedPredictions = predictedLabels
        
    maxSamplesToPlot = np.min( ( maxSamplesToPlot, embeddedData.shape[0] ))
    
    # decimate large datasets to a reasonable number of points [randomly sampled] for plotting
    if embeddedData.shape[0] > maxSamplesToPlot:        
        # read a maxSamplesToPlot number of samples (shuffled) and convert to numpy [CPU] for plotting
        targetSamples = np.random.permutation(embeddedData.shape[0])[0:maxSamplesToPlot]
        xyzViz = embeddedData.iloc[targetSamples, :].as_matrix()  # copy decimated 3D data into numpy format [ required for plotting ]
        labelsViz = decimatedLabels.iloc[targetSamples].copy()    # create a new [small] cudf dataframe of the copied decimated labels
        if decimatedPredictions is not None:
            predictionsViz = decimatedPredictions.iloc[targetSamples]
        else:
            predictionsViz = None
    else:
        # undecimated data ( presumably its small )
        xyzViz = embeddedData.as_matrix() # convert 3D data to numpy arrays required for plotting
        labelsViz = decimatedLabels
        predictionsViz = decimatedPredictions
    
    # apply colors via boolean class mapping -- equality search using cudf dataframe of decimated labels    
    nClasses = labelsViz[labelsViz.columns[0]].nunique()
    for iClass in range ( nClasses ):
        boolMask = labelsViz == iClass
        boolMask = boolMask.as_matrix()

        classColor = np.concatenate( ( np.array(matplotlib.colors.hex2color( rapidsColorsHex[iClass%nRapidsColors] )), np.array([1.,])))

        if iClass == 0:            
            colorStack = boolMask * np.ones ( (xyzViz.shape[0], 4) )  * classColor
        else:
            colorStack += boolMask * np.ones ( (xyzViz.shape[0], 4) )  * classColor
    
    if noPlotFlag:
        return xyzViz, colorStack
    
    scatterPlot = ipv.scatter ( xyzViz[:,0], xyzViz[:,1], xyzViz[:,2], 
                                    color = colorStack, marker = 'sphere', size = .5, 
                                    selected = np.arange(xyzViz.shape[0]),
                                    size_selected = 1,
                                    color_selected = colorStack )
    '''
    if predictionsViz is not None:
        npLabels = np.squeeze(labelsViz.as_matrix().astype(int))
        mispredictedSamples = np.where ( npLabels != predictionsViz )[0]        
        ipv.scatter( xyzViz[mispredictedSamples,0], 
                     xyzViz[mispredictedSamples,1], 
                     xyzViz[mispredictedSamples,2], color = [1, 1, 1, .9], marker = 'diamond', size = .5)
    '''
    return scatterPlot, colorStack
    
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

def plot_particle_evals ( swarm ):
    swarmDF, particleHistory = viz_prep( swarm )
    swarmDF.plot.bar( figsize = (figsizeWidth, figsizeHeight), 
                      color = rapidsColorsHex[1],
                      edgecolor = 'black', zorder=3, legend=[], fontsize = 15);
    plt.xlabel('pID', fontsize=15); plt.ylabel('nEvals\n', fontsize = 15);
    plt.grid(True, zorder=0)
    plt.xticks(rotation='horizontal')
    plt.title('\nEvaluations Per Particle\n', fontsize = 15);
    
def plot_boosting_rounds_histogram ( swarm ):
    pd.DataFrame( swarm.nTreesHistory ).plot.hist( figsize = (figsizeWidth, figsizeHeight), bins=15,
                                                   legend=[], zorder=3,
                                                   color = '#984dfb', 
                                                   edgecolor = 'black', fontsize = 15 );
    plt.xlabel('\nParticle Boosting Rounds [ Early Stopping ]', fontsize = 15); 
    plt.ylabel('Frequency\n', fontsize = 15);
    textStr = ' | '.join( (f'mean   : {np.mean(swarm.nTreesHistory):0.2f}',
                          f'median : {np.median(swarm.nTreesHistory):0.2f}',                    
                          f'max    : {np.max(swarm.nTreesHistory):0.2f}') )   
    
    plt.title('\nHistogram of Boosting Rounds\n\n' + textStr, fontsize=15);
    plt.grid(True, zorder=0)

def viz_particle_trails( swarm, topN = None, globalBestIndicatorFlag = True, tabbedPlot = False ):
    
    startTime = time.time()
    targetMedianParticleSize = 3
    nTreesDescaler = np.median( swarm.nTreesHistory ) / targetMedianParticleSize 
    minParticleSize = 2.5; maxParticleSize = 10;
    
    ipvFig = ipv.figure( width = ipvPlotWidth, height = ipvPlotHeight )
    
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
        
    for iParticle in range( swarm.nParticles ):

        if len( swarm.particles[ iParticle ].posHistory ):
            particlePosHistory = np.matrix( swarm.particles[ iParticle ].posHistory )
            
            # plot markers along the particle's journey over the epoch sequence
            if iParticle in topParticles:
                # draw top particles with their unique color and scaled by the number of trees
                sortedIndex = np.where(topParticles==iParticle)[0][0]
                particleColor = rapidsColorsHex[sortedIndex % nRapidsColors]
                
                for iEval in range( swarm.particles[iParticle].nEvals ):
                    particleSize = np.clip( swarm.particles[iParticle].nTreesHistory[iEval]/nTreesDescaler, minParticleSize, maxParticleSize)
                    xyz = np.matrix( particlePosHistory[iEval,:] )
                    ipv.scatter( xyz[:,0], xyz[:,1], xyz[:,2], size = particleSize,
                                 marker = 'sphere', color = particleColor, grow_limits = False )
            else:
                # draw non-top particle locations in gray
                particleColor = ( .7, .7, .7, .9 )
                xyz = np.matrix( particlePosHistory )
                ipv.scatter( xyz[:,0].squeeze(), xyz[:,1].squeeze(), xyz[:,2].squeeze(), size = 1.5,
                             marker = 'sphere', color = particleColor, grow_limits = False )
                
            # plot line trajectory [ applies to both top and non-top particles ]
            ipv.plot( particlePosHistory[:,0].squeeze(),
                      particlePosHistory[:,1].squeeze(),
                      particlePosHistory[:,2].squeeze(),  color = particleColor )

    # draw an intersecting set of lines/volumes to indicate the location of the best particle / parameter-set
    if globalBestIndicatorFlag:
        bestXYZ = np.matrix( swarm.globalBest['params'] )
        ipv.scatter( bestXYZ[:,0], bestXYZ[:,1], bestXYZ[:,2], 
                     color = rapidsColorsHex[1], 
                     size = 3, marker='box') 

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
    
    comboBox = append_label_buttons( swarm, ipvFig )
    if tabbedPlot:
        return comboBox
    
    display( comboBox )
    return None
    
def viz_top_particles ( swarm, topParticles = [1, 5, 10] ):
    tabWidget = widgets.Tab()
    tabs = []
    tabCount = 0
    for iTopParticles in topParticles:
        if swarm.nParticles >= iTopParticles:
            ipvFig = viz_particle_trails( swarm, topN = iTopParticles, tabbedPlot = True )
            tabs += [ipvFig]
            tabWidget.set_title(tabCount, f'Top-{iTopParticles} Particles')
            tabCount+=1
    
    tabWidget.children = tabs
    display(tabWidget)
    
def viz_swarm( swarm, paramLabels = False ):    
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
        
    ipvFig = ipv.figure()
    
    scatterPlots = ipv.scatter( particleXYZ[:, :, 0], 
                                particleXYZ[:, :, 1], 
                                particleXYZ[:, :, 2], 
                                marker = 'sphere', size = 5,
                                color = swarm.particleColorStack )
    
    xyzLabelsButton = widgets.Button(description="x, y, z labels")
    paramNamesLabelsButton = widgets.Button(description="parameter labels")
    
    ipv.animation_control( [ scatterPlots ] , interval = 400 )    
    ipv.xlim( swarm.paramRanges[0][1]-.5, swarm.paramRanges[0][2]+.5 )
    ipv.ylim( swarm.paramRanges[1][1]-.1, swarm.paramRanges[1][2]+.1 )
    ipv.zlim( swarm.paramRanges[2][1]-.1, swarm.paramRanges[2][2]+.1 )
    
    ipv.show()
    container = ipv.gcc()
    container.layout.align_items = 'center'
    # comboBox = append_label_buttons( swarm, ipv.gcc())#HBox([ipvFig, animControl]) )
    # display( comboBox )
    
def append_label_buttons( swarm, ipvFig ):
    
    xyzLabelsButton = widgets.Button(description="x, y, z labels")
    paramNamesLabelsButton = widgets.Button(description="parameter labels")
    
    def xyz_labels(_):
        ipv.figure(ipvFig).set_trait('xlabel','x')
        ipv.figure(ipvFig).set_trait('ylabel','y')
        ipv.figure(ipvFig).set_trait('zlabel','z')
    def param_labels(_):
        ipv.figure(ipvFig).set_trait('xlabel', str(swarm.paramRanges[0][0]))
        ipv.figure(ipvFig).set_trait('ylabel', str(swarm.paramRanges[1][0]))
        ipv.figure(ipvFig).set_trait('zlabel', str(swarm.paramRanges[2][0]))
    
    xyzLabelsButton.on_click(xyz_labels)
    paramNamesLabelsButton.on_click(param_labels)

    comboBox = VBox([ipvFig, HBox([xyzLabelsButton,paramNamesLabelsButton])] )
    comboBox.layout.align_items = 'center'
    return comboBox    

''' -------------------------------------------------------------------------
>  visualize impact of various parameters on synthetic data
------------------------------------------------------------------------- '''
def visualize_synthetic_data_variants ( coilType = 'helix', nCoils = [3, 6, 9], nSamples = 10000, coil1StDev = .3, coil2StDev= .3, decimation = 2):
    
    decimation = np.clip( decimation, 2, 20 )
    
    fig = plt.figure( figsize=(10+5*len(nCoils), 9 ))
    plt.subplots_adjust( left=0, bottom=0, right=.95, top=.95, wspace=.01, hspace=.01 )
    
    iSubplot = 1
    for iCoils in nCoils:
        xyz, colors = gen_synthetic_dataset_variant( coilType, iCoils, nSamples, coil1StDev, coil2StDev, decimation )
        
        
        ax3D = fig.add_subplot(1, len(nCoils), iSubplot, projection='3d')
        ax3D.scatter( xyz[:,0], xyz[:,1], xyz[:,2], c=colors) 
        ax3D.view_init( elev = 50, azim = 25 )
        ax3D.set_title( f'{coilType}, nCoils : {iCoils}', size=20)        
        
        ax3D.xaxis.pane.set_facecolor('w')
        ax3D.yaxis.pane.set_facecolor('w')
        ax3D.zaxis.pane.set_facecolor('w')
        #ax3D.set_facecolor('white')
        iSubplot += 1

def gen_synthetic_dataset_variant (coilType = 'helix', coils=9, nSamples = 100000, coil1StDev = .3, coil2StDev= .3, decimation = 10):
    import data_utils
    data, labels, _ = data_utils.generate_synthetic_dataset ( coilType = coilType, nSamples = nSamples, coilDensity = coils,
                                                              coil1StDev = coil1StDev, coil2StDev = coil2StDev, shuffleFlag = True)
    xyz, colors = plot_decimated_data_layer( data, labels, noPlotFlag = True )
    return xyz, colors