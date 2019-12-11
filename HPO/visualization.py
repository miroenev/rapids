import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ipywidgets import FloatSlider, ColorPicker, VBox, HBox, Label, jslink
import ipywidgets as widgets
from IPython.display import Image

import pandas as pd
import ipyvolume as ipv
import numpy as np
import cudf
import cuml
import cupy
import xgboost

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

def plot_data_train_vs_test ( dataset, vizConfig ):
    
    startTime = time.time()
    
    maxSamplesToPlot = vizConfig['maxSamplesToPlot']
    embeddingSampleLimit = vizConfig['embedding']['sampleLimit']
    dimReductionMethod = vizConfig['embedding']['method']
    
    
    if dataset.data.shape[1] > 3:
        embeddingRequiredFlag = True
    else:
        embeddingRequiredFlag = False
        
    if embeddingRequiredFlag and maxSamplesToPlot > embeddingSampleLimit:
        print(f'since embedding is necessary, further decimating to embedding sample limit : {embeddingSampleLimit} ')
        maxSamplesToPlot = embeddingSampleLimit
                        
    ipvFig = comboBox = ipv.figure ( width = ipvPlotWidth, height = ipvPlotHeight, controls=False )

    # determine split of maxSamplesToPlot per data-subset
    trainDataFraction = dataset.trainData.shape[0]/( dataset.data.shape[0] * 1.)
    testDataFraction = 1. - trainDataFraction
    testDataSamplesToPlot = int( maxSamplesToPlot * testDataFraction )
    trainDataSamplesToPlot = int( maxSamplesToPlot * trainDataFraction )
    
    
    # decimate train 
    print(f'shuffling and decimating - train-set ... ', end='')
    targetSamples = cupy.random.permutation(dataset.trainData.shape[0])[0:trainDataSamplesToPlot]
    decimatedTrainData = dataset.trainData.iloc[targetSamples, :].copy()
    decimatedTrainLabels = dataset.trainLabels.iloc[targetSamples].copy()
    
    # decimate test
    print('test-set ...')
    targetSamples = cupy.random.permutation(dataset.testData.shape[0])[0:testDataSamplesToPlot]
    decimatedTestData = dataset.testData.iloc[targetSamples, :].copy()
    decimatedTestLabels = dataset.testLabels.iloc[targetSamples].copy()
    
    # embed train and test
    if embeddingRequiredFlag: 
        print('embedding train-set into new shape ', end='')
        embeddedTrain, embeddingModel = reduce_to_3D ( decimatedTrainData, decimatedTrainLabels, dimReductionMethod )
        
        print('embedding test-set into new shape ', end='')
        embeddedTest, _ = reduce_to_3D ( decimatedTestData, decimatedTestLabels, dimReductionMethod, trainedEmbeddingModel = embeddingModel )
    else:
        embeddedTrain = decimatedTrainData
        embeddedTest = decimatedTestData
    
    # plot train [ possibly embed ]
    testDataScatterPlot, testColorStack = plot_decimated_data_layer( embeddedTest, decimatedTestLabels )
    # plot test [ use trained embedding model ]
    trainDataScatterPlot, trainColorStack = plot_decimated_data_layer( embeddedTrain, decimatedTrainLabels)    
    
    plot_feature_distributions ( embeddedTrain, embeddedTest ) # saves file to local storage
    
    
    # sliders to impact marker size of train and test data
    testSize = FloatSlider( min=0.1, max=5., step=0.25, readout_format='.1f')
    jslink((testDataScatterPlot, 'size_selected'), (testSize, 'value'))

    trainSize = FloatSlider(min=0.1, max=5., step=0.25, readout_format='.1f')
    jslink((trainDataScatterPlot, 'size_selected'), (trainSize, 'value'))
        
    # buttons to allow coloring test and train data and reverting back
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
        
    htmlString = f"<font size=5>Dataset : " \
                 f"<font color='{rapidsColorsHex[0]}'>{dataset.config['datasetName'].capitalize()}" \
                 "</font></font>"
    
    tabWidget = widgets.Tab()
    
    fDistroImage = HBox( [ widgets.Image( value = open('feature_distributions.png', 'rb').read(), 
                                            format = 'png' ) ] )
    fDistroImage.layout.justify_content='center'
    
    comboBox = VBox( [ ipvFig, widgets.HTML(value=htmlString), 
                                HBox( [ Label('TEST  marker size:'), testSize]), 
                                HBox( [ Label('TRAIN marker size:'), trainSize]), 
                                HBox( [ markTestButton, unmarkTestButton ] ), 
                                HBox( [ markTrainButton, unmarkTrainButton ] ) ] )        
    
    comboBox.layout.align_items = 'center'
    tabWidget.set_title(0, f'Interactive')
    tabWidget.set_title(1, f'Distributions')
    tabWidget.children = [ comboBox, fDistroImage ]
    
    print(f'elapsed time : {time.time()- startTime}')
    display(tabWidget)
    
    
def plot_decimated_data_layer ( data, labels, noPlotFlag = False):

    # apply colors via boolean class mapping -- equality search using cudf dataframe of decimated labels    
    nClasses = labels[labels.columns[0]].nunique()
    
    nSamples = data.shape[0]
    for iClass in range ( nClasses ):
        boolMask = labels == iClass
        boolMask = boolMask.as_matrix()

        classColor = np.concatenate( ( np.array(matplotlib.colors.hex2color( rapidsColorsHex[iClass%nRapidsColors] )), np.array([1.,])))

        if iClass == 0:            
            colorStack = boolMask * np.ones ( (nSamples, 4) )  * classColor
        else:
            colorStack += boolMask * np.ones ( (nSamples, 4) )  * classColor
    
    xyzViz = data.as_matrix()
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
>  dataset plotting and dimensionality reduction
>  train and test set visualization with interactive sliders/buttons
------------------------------------------------------------------------- '''
def reduce_to_3D ( data, labels, dimReductionMethod, trainedEmbeddingModel = None ):
    
    startTime = time.time()        
    
    preTrainedStr = ''

    '''
    if dimReductionMethod == 'TSNE':        
        embeddingModel = None
        embeddedData = cuml.TSNE( n_components = 2 ).fit_transform ( X = data )
        embeddedData.add_column('3', cudf.Series(np.zeros((data.shape[0]))) )
    else:
    '''
    if trainedEmbeddingModel is not None:
        preTrainedStr = 'pre-trained '
        embeddingModel = trainedEmbeddingModel
    else:
        if dimReductionMethod == 'PCA':        
            embeddingModel = cuml.PCA( copy = True, n_components = 3,
                                       random_state = 0,
                                       svd_solver = 'full',
                                       verbose = True, 
                                       whiten = False ).fit( X = data )

        elif dimReductionMethod == 'UMAP':
            embeddingModel = cuml.UMAP( n_components = 3 ).fit( X = data, y = labels )                
        else:
            assert('unable to find embedding model match to user query')
        
    embeddedData = embeddingModel.transform ( X = data )            
   
    elapsedTime = time.time() - startTime
    print(f'{embeddedData.shape} via {preTrainedStr}{dimReductionMethod} -- completed in: {elapsedTime:.3f} seconds')

    return embeddedData, embeddingModel

def plot_feature_distributions ( embeddedTrain, embeddedTest, maxSamplesToPlot = 1000, nBins = 30 ):
    fig = plt.figure(figsize=(10,10))
    plt.subplots_adjust(left=0.1, right=.9, bottom=0.1, top=.9,  wspace=.1, hspace=.5)
    
    startTime = time.time()
    
    featureColumns = embeddedTrain.columns
    
    nColumns = len( featureColumns ) 
    for iColumn in range( nColumns ) :
        iColumnName = featureColumns[iColumn]
        ax = plt.subplot( nColumns, 1, iColumn + 1 )

        upperBound = max( embeddedTrain[iColumnName].max(), embeddedTest[iColumnName].max() ) 
        lowerBound = max( embeddedTrain[iColumnName].min(), embeddedTest[iColumnName].min() )     
        trainDataNP = np.hstack( [lowerBound,  embeddedTrain[iColumnName].to_array(), upperBound])
        testDataNP = np.hstack( [lowerBound,  embeddedTest[iColumnName].to_array(), upperBound])    
        ax.hist(trainDataNP, bins = nBins, color = '#666666', alpha = 1)
        ax.hist(testDataNP, bins = nBins, color = '#ffb500', alpha = 0.75)
        ax.legend(['train', 'test'])
        ax.set_title(str(iColumnName), fontsize=15)
    
    plt.savefig('feature_distributions.png')
    plt.close(fig);
    
    print(f'computing feature distributions in {time.time() - startTime:0.2f} seconds')    
    
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
    
def viz_particle_trails ( swarm, topN = None, globalBestIndicatorFlag = True, tabbedPlot = False ):    
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
            
    sortedParticles = pd.DataFrame.from_dict(particleMaxes, orient='index').sort_values(by=0, ascending=False)
    topParticles = sortedParticles.index[0:np.min((np.max((topN,1)),swarm.nParticles))]
        
    for iParticle in range( swarm.nParticles ):

        if len( swarm.particles[ iParticle ].posHistory ):
            
            particlePosHistory = np.matrix( swarm.particles[ iParticle ].posHistory )
            
            # plot markers along the particle's journey over the epoch sequence
            if iParticle in topParticles:
                
                # draw top particles with their unique color and scaled by the number of trees
                sortedIndex = np.where(topParticles==iParticle)[0][0]
                particleColor = rapidsColorsHex[sortedIndex % nRapidsColors]
                particleSizes = {}
                for iEval in range( swarm.particles[iParticle].nEvals ):
                    particleSizes[iEval] = np.clip( swarm.particles[iParticle].nTreesHistory[iEval]/nTreesDescaler, minParticleSize, maxParticleSize)
                
                #import pdb; pdb.set_trace()                
                ipv.scatter( particlePosHistory[:,0].squeeze(),
                             particlePosHistory[:,1].squeeze(),
                             particlePosHistory[:,2].squeeze(), 
                             size = np.array(list(particleSizes.values())),
                             marker = 'sphere', color = particleColor, grow_limits = False )

            else:
                # draw non-top particle locations in gray
                particleColor = ( .7, .7, .7, .9 )
                ipv.scatter( particlePosHistory[:,0].squeeze(),
                             particlePosHistory[:,1].squeeze(),
                             particlePosHistory[:,2].squeeze(), size = 1.5,
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

        data = np.ones((100,2,2))
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
        
    ipvFig = ipv.figure( width = ipvPlotWidth, height = ipvPlotHeight )
    
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
        
    container = ipv.gcc()
    container.layout.align_items = 'center'
    comboBox = append_label_buttons( swarm, ipvFig, container )    
    display( comboBox )
    
def append_label_buttons( swarm, ipvFig, container=None ):
    
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
    
    if container is None:
        container = ipvFig
    comboBox = VBox([container, HBox([xyzLabelsButton,paramNamesLabelsButton])] )
    comboBox.layout.align_items = 'center'
    return comboBox    

''' -------------------------------------------------------------------------
>  xgboost model visualization tools
------------------------------------------------------------------------- '''

def plot_first_N_trees ( trainedModel, nTrees = 5, figSize = None ):
    df = trainedModel.trees_to_dataframe()
    nodesPerTree = int( df['Tree'].value_counts().mean() )
    
    if figSize is None:
        figSize = ( np.clip( nTrees * nodesPerTree * 3, 25, 2**16), nodesPerTree//2 )
        print(figSize)
        
    plt.figure ( figsize = figSize, facecolor=(1, 1, 1) )
    for iTree in range(nTrees):
        ax = plt.subplot(1, nTrees, iTree + 1)
        xgboost.plot_tree ( trainedModel, num_trees = iTree, ax = ax )

def plot_feature_importance( trainedModel, maxFeatures = 10, color='#a788e4'):
    plt.figure ( figsize = (10,7) )
    ax = plt.subplot(1,1,1)
    xgboost.plot_importance ( trainedModel, 
                              max_num_features = maxFeatures, 
                              color=color, ax = ax, zorder=3 );
    plt.grid(True, zorder=0)
    
    
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
    