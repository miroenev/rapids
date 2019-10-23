import ipyvolume as ipv
import numpy as np
import cudf
import cuml
import time

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
    labelsCopy[ labelsCopy != 1 ] = -1
    classMask_1 = labelsCopy[ labelsCopy == 1].fillna(0).as_matrix()
    classMask_2 = -1 * labelsCopy[ labelsCopy == -1].fillna(0).as_matrix()

    color = classMask_1 * np.ones ( (xyzViz.shape[0], 4) )  * [ 116/255., 0/255., 255/255., 1] + \
            classMask_2 * np.ones ( (xyzViz.shape[0], 4) )  * [ 100/255., 225/255., 0., 1]
    
    ipv.figure()
    ipv.scatter( xyzViz[:,0], xyzViz[:,1], xyzViz[:,2], color = color, marker = 'sphere', size = .5)
    ipv.pylab.squarelim()
    ipv.show()