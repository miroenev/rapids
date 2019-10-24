import ipyvolume as ipv
import numpy as np
import cudf
import cuml
import time


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

def reduce_to_3D ( data, labels, datasetName, dimReductionMethod = 'PCA', maxSamplesForDimReduction = None ):
    startTime = time.time()        
    
    # decimate to limit dim-reduction out of memory errors
    if maxSamplesForDimReduction is None:
        decimationFactor = 1        
    else:        
        decimationFactor = np.max((1, data.shape[0] // maxSamplesForDimReduction))
    
    if datasetName == 'synthetic':   
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
                
    elapsedTime = time.time() - startTime
    print(f'new shape: {embeddedData.shape} -- completed in: {elapsedTime:.3f} seconds')
    return embeddedData, decimatedLabels

def plot_data( data, labels, datasetName, dimReductionMethod = 'PCA', 
               maxSamplesForDimReduction = 1000000, 
               maxSamplesToPlot = 100000 ):
    '''
    assumes binary labels
    '''
    
    print(f'plotting {datasetName.upper()} dataset, original shape: {data.shape}')
    if data.shape[1] != 3:
        embeddedData, decimatedLabels = reduce_to_3D ( data, labels, datasetName, dimReductionMethod, maxSamplesForDimReduction )
    else:
        embeddedData = data
        decimatedLabels = labels
    
    # decimation for visualizaiton
    maxSamplesToPlot = np.min( ( maxSamplesToPlot, embeddedData.shape[0] ))
        
    if embeddedData.shape[0] != maxSamplesToPlot:
        percentOfTotal = maxSamplesToPlot/embeddedData.shape[0] * 100.
        print( f' > plotting subset of {maxSamplesToPlot} samples -- {percentOfTotal}% of total, adjust via maxSamplesToPlot ')
        
        # read a maxSamplesToPlot number of samples (shuffled) and convert to numpy [CPU] for plotting
        targetSamples = np.random.permutation(embeddedData.shape[0])[0:maxSamplesToPlot]
        xyzViz = embeddedData.iloc[targetSamples, :].as_matrix()  # note implicit copy in the cudf->numpy conversion
        labelsCopy = decimatedLabels.iloc[targetSamples].copy()
    else:
        xyzViz = embeddedData.as_matrix()
        labelsCopy = decimatedLabels.copy()
    
    # apply colors   
    nClasses = labelsCopy[labelsCopy.columns[0]].nunique()
    
    for iClass in range ( nClasses ):
        boolMask = labelsCopy == iClass
        boolMask = boolMask.as_matrix()
        
        if iClass == 0:
            color = boolMask * np.ones ( (xyzViz.shape[0], 4) )  * rapidsColors[iClass]
        else:
            color += boolMask * np.ones ( (xyzViz.shape[0], 4) )  * rapidsColors[iClass]
    
    ipv.figure()
    ipv.scatter( xyzViz[:,0], xyzViz[:,1], xyzViz[:,2], color = color, marker = 'sphere', size = .5)
    ipv.pylab.squarelim()
    ipv.show()