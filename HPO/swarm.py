import numpy as np
import pandas as pd

import time
import copy

import cupy
import cudf

import dask
from dask import delayed
from dask.distributed import as_completed
import xgboost

import matplotlib
import visualization as viz

''' ---------------------------------------------------------------------
>  HPO / PARTICLE SWARM
----------------------------------------------------------------------'''
def sample_params (paramRanges, randomSeed = None):
    if randomSeed is not None:
        np.random.seed(randomSeed)
    
    paramSamples = []
    velocitySamples = []
    
    for iParam, paramRange in paramRanges.items():        
        lowerBound = paramRange[1]
        upperBound = paramRange[2]
        paramType = paramRange[3]

        if paramType == 'int':
            paramSamples += [ np.random.randint( low = lowerBound, high = upperBound) ]
            
        elif paramType =='float':            
            paramSamples += [ np.random.uniform( low = lowerBound, high = upperBound) ]
                
        velocitySamples += [ np.random.uniform( low = -np.abs( upperBound - lowerBound ), 
                                                high = np.abs( upperBound - lowerBound ) ) ]
        
    return np.array(paramSamples), np.array( velocitySamples)

class Particle():
    def __init__ (self, pos, velo, particleID = -1 ):
        self.pos = pos
        self.velo = velo
        self.pID = particleID
        self.personalBestParams = pos
        self.personalBestPerf = 0
        self.posHistory = []
        self.nTreesHistory = []
        self.veloHistory = []
        self.evalTimeHistory = []        
        self.trainDataPerfHistory = []
        self.testDataPerfHistory = []
        self.nEvals = 0
        self.color = matplotlib.colors.hex2color( viz.rapidsColorsHex[ particleID % viz.nRapidsColors ])

def parse_tunable_params ( modelConfig ):
    paramRanges = {}
    paramCount = 0
    for key, value in modelConfig.items():
        if 'tunableParam' in key:        
            paramRanges.update( { paramCount : value } )
            paramCount += 1    
    return paramRanges 
    
class Swarm():
    def __init__ ( self, client, dataset, HPOConfig, modelConfig, computeConfig ):
        
        self.client = client
        self.dataset = dataset
        
        
        self.nParticles = HPOConfig['nParticles']
        self.nEpochs = HPOConfig['nEpochs']
        self.paramRanges = parse_tunable_params( modelConfig )

        # CPU baseline options
        if computeConfig['clusterType'] == 'LocalCluster':
            self.cpuFlag = True
            self.dataset.cpuDataset = dataset_to_CPU ( dataset )
        else:
            self.cpuFlag = False
            self.dataset.cpuDataset = None
        
        swarmName = str(type(self)).split('.')[1].strip("'>'")
        print( f'! initializing {swarmName}, with {self.nParticles} particles, and {self.nEpochs} epochs')
        self.reset_swarm()
        
    def reset_swarm( self ):
        self.swarmEvals = 0
        
        self.particles = {}
        self.delayedEvalParticles = []
        self.nTreesHistory = []
        self.globalBest = {'accuracy': 0, 'particleID': -1, 'params': [], 'nTrees': -1, 'iEvaluation': - 1}
    
    def scatter_data_to_workers( self ):
        self.scatteredDataFutures = None
        if self.client is not None:
            if not self.cpuFlag:
                self.scatteredDataFutures = self.client.scatter( [ self.dataset.trainData, self.dataset.trainLabels,
                                                                   self.dataset.testData,  self.dataset.testLabels ],
                                                                   broadcast = True )                
            else:
                self.scatteredDataFutures = self.client.scatter( [ self.dataset.cpuDataset['trainData'], 
                                                                   self.dataset.cpuDataset['trainLabels'],
                                                                   self.dataset.cpuDataset['testData'], 
                                                                   self.dataset.cpuDataset['testLabels'] ], 
                                                                   broadcast = True )
    
    def build_initial_particles( self ):
        self.delayedEvalParticles = []
        self.particleColorStack = []
        
        print(f'\n   pID |{self.paramRanges[0][0]:>15},{self.paramRanges[1][0]:>15},{self.paramRanges[2][0]:>15}')
        for iParticle in range( self.nParticles ):
            pos, velo = sample_params ( self.paramRanges, randomSeed = iParticle )
            
            # fix first and last particle to capture upper and lower bounds
            if iParticle == 0: pos[0] = self.paramRanges[0][1]; pos[1] = self.paramRanges[1][1]; pos[2] = self.paramRanges[2][1]
            if iParticle == self.nParticles-1: pos[0] = self.paramRanges[0][2]; pos[1] = self.paramRanges[1][2]; pos[2] = self.paramRanges[2][2]
                
            self.particles[iParticle] = Particle( pos, velo, particleID = iParticle )
            
            if iParticle == 0: self.particleColorStack = self.particles[iParticle].color                
            else: self.particleColorStack = np.vstack( ( self.particleColorStack, self.particles[iParticle].color ) )

            # print initial particle locations            
            print(f'   {iParticle:>3} |{pos[0]:>15.0f},{pos[1]:>15.2f},{pos[2]:>15.2f} ')
            self.delayedEvalParticles.append ( delayed ( evaluate_particle ) ( self.scatteredDataFutures,
                                                                               self.particles[iParticle].pos,
                                                                               self.paramRanges,
                                                                               self.particles[iParticle].pID,
                                                                               self.dataset.trainObjective,
                                                                               cpuFlag = self.cpuFlag ) )
        print('')
        
    def enforce_bounds( self, newParameters ):
        for iParameter in range( len ( newParameters )):
            newParameters[iParameter] = np.clip ( newParameters[iParameter], 
                                                 self.paramRanges[iParameter][1], 
                                                 self.paramRanges[iParameter][2])
        return newParameters
    
    def update_global_best ( self, latestTestDataPerf, pID, nTrees ):
        if latestTestDataPerf > self.globalBest['accuracy']:
            self.globalBest['accuracy'] = latestTestDataPerf
            self.globalBest['params'] = self.particles[pID].pos.copy()
            self.globalBest['nTrees'] = nTrees
            self.globalBest['particleID'] = pID
            self.globalBest['iEvaluation'] = self.particles[pID].nEvals
            print(f'new best {latestTestDataPerf:0.5f} found by particle {pID} on eval {self.swarmEvals}')
        
    
    def update_particle (self, latestTestDataPerf, pID, nTrees, evalTime, 
                         wMomentum = .1, wGlobalBest = .45, wPersonalBest = .35, wExplore = .1,
                         randomSeed = None):        
        if randomSeed is not None:
            np.random.seed(randomSeed)
        
        # update swarm/global best
        self.update_global_best ( latestTestDataPerf, pID, nTrees )
        
        # update particle's personal best
        if latestTestDataPerf > self.particles[pID].personalBestPerf:
            self.particles[pID].personalBestPerf = latestTestDataPerf
            self.particles[pID].personalBestParams = self.particles[pID].pos.copy()
            
        # bookeeping for early stopping number of boosting rounds
        self.nTreesHistory.append(nTrees)
        
        # computing velocity update terms [ attraction to global and personal best ]        
        socialInfluence = ( self.globalBest['params'] - self.particles[pID].pos )
        individualInfluence = ( self.particles[pID].personalBestParams - self.particles[pID].pos )
        inertiaInfluence = self.particles[pID].velo.copy()
        
        # optional random exploration term
        if wExplore > 0:  
            _, exploreVelo = sample_params (self.paramRanges)
        else: 
            exploreVelo = np.array( [0., 0., 0.] )        
        
        # compute new velocity 
        self.particles[pID].velo  =    wMomentum      *  inertiaInfluence \
                                     + wPersonalBest  *  individualInfluence  * np.random.random() \
                                     + wGlobalBest    *  socialInfluence      * np.random.random() \
                                     + wExplore * exploreVelo
        
        # apply velocity to determine new particle position
        self.particles[pID].pos = self.particles[pID].pos.copy() + self.particles[pID].velo
        
        # make sure the particle does not leave the search boundaries
        self.particles[pID].pos = self.enforce_bounds( self.particles[pID].pos )
        
    def log_particle_history ( self, testDataPerf, trainDataPerf, pID, nTrees, evalTime ):
        self.particles[pID].posHistory.append( self.particles[pID].pos )
        self.particles[pID].nTreesHistory.append( nTrees )
        self.particles[pID].veloHistory.append( self.particles[pID].velo )        
        self.particles[pID].evalTimeHistory.append( evalTime )
        self.particles[pID].trainDataPerfHistory.append( trainDataPerf )
        self.particles[pID].testDataPerfHistory.append( testDataPerf )
        self.particles[pID].nEvals += 1
        
    def report_final_params ( self ):
        print('search completed...\n\n')
        print(f"{'accuracy':>15} : {self.globalBest['accuracy']:0.5f}")
        print(f"{'elapsed time':>15} : {self.elapsedTime:0.3f} seconds")

        
        print(f"\n{'parameter':>15} | opt. value")
        print('----------------------------------')
        print(f"{self.paramRanges[0][0]:>15} : {int(self.globalBest['params'][0])}")
        print(f"{self.paramRanges[1][0]:>15} : {self.globalBest['params'][1]:0.3f}")
        print(f"{self.paramRanges[2][0]:>15} : {self.globalBest['params'][2]:0.3f}")
        print(f"{'nTrees':>15} : {self.globalBest['nTrees']}")        
        
        
class SyncSwarm ( Swarm ):
    def run_search( self, asyncInitializeFlag = False ):
        startTime = time.time()
        
        self.reset_swarm ()
        self.scatter_data_to_workers ()
        self.build_initial_particles ()
        
        if asyncInitializeFlag: epochsToRun = 1
        else: epochsToRun = self.nEpochs
        
        for iEpoch in range( epochsToRun ):
            futureEvalParticles = self.client.compute( self.delayedEvalParticles )
            self.delayedEvalParticles = []
            
            for iParticleFuture in futureEvalParticles:
                
                testDataPerf, trainDataPerf, pID, nTrees, evalTime = iParticleFuture.result()
                
                self.log_particle_history ( testDataPerf, trainDataPerf, pID, nTrees, evalTime )
                self.update_particle ( testDataPerf, pID, nTrees, evalTime, 
                                       randomSeed = pID + self.swarmEvals, wExplore = 0 )
                
                self.swarmEvals += 1
                
                self.delayedEvalParticles.append ( delayed ( evaluate_particle ) ( self.scatteredDataFutures,
                                                                                   self.particles[pID].pos,
                                                                                   self.paramRanges,
                                                                                   self.particles[pID].pID,
                                                                                   self.dataset.trainObjective,
                                                                                   cpuFlag = self.cpuFlag ) )
            print(f'> sync epoch {iEpoch} of {epochsToRun}')
        self.elapsedTime = time.time() - startTime
        if not asyncInitializeFlag: self.report_final_params()
        
# see dask documentation https://docs.dask.org/en/latest/futures.html#distributed.as_completed
class AsyncSwarm ( SyncSwarm ):
    def run_search( self, syncWarmupFlag = False ):
        startTime = time.time()
        
        if syncWarmupFlag:
            print('sync warmup')
            super().run_search( asyncInitializeFlag = True)
            print('sync warmup complete\n')
            print('continuing with async search')
        else:
            self.reset_swarm ()
            self.scatter_data_to_workers ()
            self.build_initial_particles ()
        
        futureEvalParticles = self.client.compute( self.delayedEvalParticles )
        particleFutureSeq = as_completed( futureEvalParticles )
        
        # particleFutureSeq is an iterator of futures, to which we append newly updated particles   
        for particleFuture in particleFutureSeq:
            testDataPerf, trainDataPerf, pID, nTrees, evalTime = particleFuture.result()
            
            self.log_particle_history( testDataPerf, trainDataPerf, pID, nTrees, evalTime )
            self.update_particle ( testDataPerf, pID, nTrees, evalTime, wExplore =  0 )

            self.swarmEvals += 1

            # termination condition
            approximateEpoch = self.swarmEvals // self.nParticles
            if approximateEpoch > self.nEpochs: break
            
            # create delayed evaluations for newly updated particles 
            delayedParticle = delayed ( evaluate_particle ) ( self.scatteredDataFutures,
                                                              self.particles[pID].pos,
                                                              self.paramRanges,
                                                              self.particles[pID].pID,
                                                              self.dataset.trainObjective,
                                                              cpuFlag = self.cpuFlag )
            
            futureParticle = self.client.compute( delayedParticle )
            particleFutureSeq.add( futureParticle )
            
            # print progress update via approximate epoch
            if self.swarmEvals % self.nParticles == 0:
                print(f'> async epoch {approximateEpoch} of {self.nEpochs}')

        self.elapsedTime = time.time() - startTime
        self.report_final_params()

class RandomSearchAsync ( AsyncSwarm ):
    def update_particle ( self, latestTestDataPerf, pID, nTrees, evalTime, wExplore = 1 ):
        # update swarm/global best
        self.update_global_best( latestTestDataPerf, pID, nTrees )   

        # bookeeping for early stopping number of boosting rounds
        self.nTreesHistory.append(nTrees)
        
        # apply velocity to determine new particle position
        self.particles[pID].pos, _ = sample_params (self.paramRanges)
              
# xgboost parameters -- https://xgboost.readthedocs.io/en/latest/parameter.html
def evaluate_particle ( scatteredDataFutures, particleParams, paramRanges, particleID, 
                        trainObjective, earlyStoppingRounds = 250, cpuFlag = False, nCPUWorkers = 1 ) :
    
    trainDataFuture = scatteredDataFutures[0]
    trainLabelsFuture = scatteredDataFutures[1]
    testDataFuture = scatteredDataFutures[2]
    testLabelsFuture = scatteredDataFutures[3]
    
    if not cpuFlag:
        xgboostParams = {
            'tree_method'  : 'gpu_hist',
            'random_state' : 0, 
        }        
    else:
        xgboostParams = {
            'tree_method'  : 'hist',
            'n_jobs'       : nCPUWorkers,
            'random_state' : 0, 
        }
    
    # objective [ binary or multi-class ]
    xgboostParams['objective'] = trainObjective[0]
    if trainObjective[1] is not None: xgboostParams['num_class'] = trainObjective[1]
    
    def enforce_type( parameterValue, paramRange ):        
        if paramRange[3] == 'int': 
            return int( parameterValue )
        elif paramRange[3] == 'float':
            return float( parameterValue )
        
    # flexible parameters
    xgboostParams[paramRanges[0][0]] = enforce_type( particleParams[0], paramRanges[0] )
    xgboostParams[paramRanges[1][0]] = enforce_type( particleParams[1], paramRanges[1] )
    xgboostParams[paramRanges[2][0]] = enforce_type( particleParams[2], paramRanges[2] )
    xgboostParams['num_boost_rounds'] = 2000
    
    startTime = time.time()
        
    trainDMatrix = xgboost.DMatrix( data = trainDataFuture, label = trainLabelsFuture )
    testDMatrix = xgboost.DMatrix( data = testDataFuture, label = testLabelsFuture )
        
    trainedModelGPU = xgboost.train( dtrain = trainDMatrix, evals = [(testDMatrix, 'test')], 
                                     params = xgboostParams,
                                     num_boost_round = xgboostParams['num_boost_rounds'], 
                                     early_stopping_rounds = earlyStoppingRounds,
                                     verbose_eval = False )

    trainDataPerf = 1 - float( trainedModelGPU.eval(trainDMatrix).split(':')[1] )
    testDataPerf = 1 - float( trainedModelGPU.eval(testDMatrix).split(':')[1] )   
    
    nTrees = trainedModelGPU.best_iteration

    elapsedTime = time.time() - startTime

    return testDataPerf, trainDataPerf, particleID, nTrees, elapsedTime

def dataset_to_CPU ( dataset ):
    cpuDataset = {}
    cpuDataset['trainData'] = dataset.trainData.to_pandas().values
    cpuDataset['testData'] = dataset.testData.to_pandas().values
    cpuDataset['trainLabels'] = dataset.trainLabels.to_pandas().values
    cpuDataset['testLabels'] = dataset.testLabels.to_pandas().values
    return cpuDataset

import pprint
def evaluate_manual_params (dataset, manualXGBoostParams):
    startTime = time.time()

    manualXGBoostParams['objective'] = dataset.trainObjective[0]
    if dataset.trainObjective[1] is not None: manualXGBoostParams['num_class'] = dataset.trainObjective[1]    
    
    trainDMatrix = xgboost.DMatrix( data = dataset.trainData, label = dataset.trainLabels )
    trainedModelGPU = xgboost.train( dtrain = trainDMatrix, 
                                     params = manualXGBoostParams, 
                                     num_boost_round = manualXGBoostParams['num_boost_round'] )
    
    trainTime = time.time() - startTime
    
    startTime = time.time()
    
    testDMatrix = xgboost.DMatrix( data = dataset.testData, label = dataset.testLabels )    

    trainAccuracy = 1 - float( trainedModelGPU.eval(trainDMatrix).split(':')[1] )
    testAccuracy = 1 - float( trainedModelGPU.eval(testDMatrix).split(':')[1] )   

    inferenceTime = time.time() - startTime
    
    print(f'train accuracy : {trainAccuracy:0.4f} in {trainTime:0.4f} seconds')
    print(f'test  accuracy : {testAccuracy:0.4f} in {inferenceTime:0.4f} seconds')
    return trainedModelGPU, trainTime, inferenceTime
    
def evaluate_manual_params_CPU ( dataset, manualXGBoostParamsCPU, nCPUWorkers = 1 ):
    startTime = time.time()
    
    datasetCPU = dataset_to_CPU ( dataset )
    manualXGBoostParamsCPU['tree_method'] = 'hist'
    manualXGBoostParamsCPU['n_jobs'] = nCPUWorkers

    # inherit objective
    manualXGBoostParamsCPU['objective'] = dataset.trainObjective[0]
    if dataset.trainObjective[1] is not None: manualXGBoostParamsCPU['num_class'] = dataset.trainObjective[1]                    
    
    trainDMatrixCPU = xgboost.DMatrix( data = datasetCPU['trainData'], 
                                       label = datasetCPU['trainLabels'] )
    
    trainedModelCPU = xgboost.train( dtrain = trainDMatrixCPU, 
                                     params = manualXGBoostParamsCPU, 
                                     num_boost_round = manualXGBoostParamsCPU['num_boost_round'] )
    
    trainTime = time.time() - startTime
    
    startTime = time.time()
    testDMatrixCPU = xgboost.DMatrix( data = datasetCPU['testData'], 
                                      label = datasetCPU['testLabels'])
    
    trainAccuracy = 1 - float( trainedModelCPU.eval(trainDMatrixCPU).split(':')[1] )
    testAccuracy = 1 - float( trainedModelCPU.eval(testDMatrixCPU).split(':')[1] )   

    inferenceTime = time.time() - startTime
    print(f'train accuracy : {trainAccuracy:0.4f} in {trainTime:0.4f} seconds')
    print(f'test  accuracy : {testAccuracy:0.4f} in {inferenceTime:0.4f} seconds')
    return trainedModelCPU, trainTime, inferenceTime