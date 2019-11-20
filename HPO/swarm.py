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

''' ---------------------------------------------------------------------
>  HPO / PARTICLE SWARM
----------------------------------------------------------------------'''
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

class Particle():
    def __init__ (self, pos, velo, particleID = -1 ):
        self.pos = pos
        self.velo = velo
        self.pID = particleID
        self.personalBestParams = pos
        self.personalBestPerf = 0
        self.posHistory = []
        self.evalTimeHistory = []
        self.nEvals = 0
        
class Swarm():
    def __init__ ( self, client, dataset, paramRanges, nParticles = 10, nEpochs = 10 ):
        
        self.client = client
        self.dataset = dataset
        self.paramRanges = paramRanges
        
        self.nParticles = nParticles
        self.nEpochs = nEpochs
        self.reset_swarm()
        
    def reset_swarm( self ):
        self.nEvals = 0
        
        self.particles = {}
        self.delayedEvalParticles = []

        self.globalBest = {'accuracy': 0, 'particleID': -1, 'params': [], 'iEvaluation': - 1}
    
    def scatter_data_to_workers( self ):
        self.scatteredDataFutures = None
        if self.client is not None:
            self.scatteredDataFutures = self.client.scatter( [ self.dataset.trainData, self.dataset.trainLabels,
                                                               self.dataset.testData,  self.dataset.testLabels ], broadcast = True )
    
    def build_initial_particles( self ):
        self.delayedEvalParticles = []
        for iParticle in range( self.nParticles ):
            pos, velo = sample_params ( self.paramRanges )
            self.particles[iParticle] = Particle( pos, velo, particleID = iParticle )
            self.delayedEvalParticles.append ( delayed ( evaluate_particle ) ( self.scatteredDataFutures,
                                                                               self.particles[iParticle].pos,
                                                                               self.paramRanges,
                                                                               self.particles[iParticle].pID,
                                                                               self.dataset.trainObjective ) )

    def enforce_bounds( self, newParameters ):
        for iParameter in range( len ( newParameters )):
            newParameters[iParameter] = np.clip ( newParameters[iParameter], self.paramRanges[iParameter][1], self.paramRanges[iParameter][2])
        return newParameters
    
    def update_particle (self, pID, latestTestDataPerf, evalTime, 
                         wMomentum = .1, wGlobalBest = .55, wPersonalBest = .35, 
                         mode = 'classification'):
        
        if latestTestDataPerf > self.globalBest['accuracy']:
            self.globalBest['accuracy'] = latestTestDataPerf
            self.globalBest['params'] = self.particles[pID].pos.copy()
            self.globalBest['particleID'] = pID
            print(f'new global best {latestTestDataPerf:0.5f} found by particle {pID}, at eval {self.nEvals}')
        
        if latestTestDataPerf > self.particles[pID].personalBestPerf:
            self.particles[pID].personalBestPerf = latestTestDataPerf
            self.particles[pID].personalBestParams = self.particles[pID].pos.copy()
            print(f'\t\t new personal best {latestTestDataPerf:0.5f} found by particle {pID}, at eval {self.nEvals}')
        
        # computing update terms for particle swarm
        inertiaInfluence = self.particles[pID].velo.copy()
        socialInfluence = ( self.globalBest['params'] - self.particles[pID].pos )
        individualInfluence = ( self.particles[pID].personalBestParams - self.particles[pID].pos )

        self.particles[pID].velo  =    wMomentum      *  inertiaInfluence     \
                                     + wPersonalBest  *  individualInfluence  * np.random.random()   \
                                     + wGlobalBest    *  socialInfluence      * np.random.random()

        self.particles[pID].pos = self.particles[pID].pos.copy() + self.particles[pID].velo
        self.particles[pID].pos = self.enforce_bounds( self.particles[pID].pos )
        
        self.particles[pID].posHistory.append( self.particles[pID].pos )
        self.particles[pID].nEvals += 1
        self.particles[pID].evalTimeHistory.append( evalTime )
        
class SyncSwarm ( Swarm ):
    def run_search( self ):
        self.reset_swarm ()
        self.scatter_data_to_workers ()
        self.build_initial_particles ()
        
        for iEpoch in range( self.nEpochs ):
            futureEvalParticles = self.client.compute( self.delayedEvalParticles )
            self.delayedEvalParticles = []
            
            for iParticleFuture in futureEvalParticles:
                testDataPerf, trainDataPerf, pID, evalTime = iParticleFuture.result()
                self.update_particle ( pID, testDataPerf, evalTime ) # inplace update to particle.pos, particle.velo
                
                self.nEvals += 1
                
                self.delayedEvalParticles.append ( delayed ( evaluate_particle ) ( self.scatteredDataFutures,
                                                                                   self.particles[pID].pos,
                                                                                   self.paramRanges,
                                                                                   self.particles[pID].pID,
                                                                                   self.dataset.trainObjective ) )
class AsyncSwarm ( Swarm ):
    def run_search( self ):
        self.reset_swarm ()
        self.scatter_data_to_workers ()
        self.build_initial_particles ()

        futureEvalParticles = self.client.compute( self.delayedEvalParticles )
        particleFutureSeq = as_completed( futureEvalParticles )
        
        # note that the particleFutureSeq is an iterator of futures
        # at the end of the loop we create new work and append it to the iterartor, this behavior resembles a while loop
        # see dask documentation https://docs.dask.org/en/latest/futures.html#distributed.as_completed
        for particleFuture in particleFutureSeq: 
            testDataPerf, trainDataPerf, pID, evalTime = particleFuture.result()
            self.update_particle ( pID, testDataPerf, evalTime ) # inplace update to particle.pos, particle.velo

            self.nEvals += 1
            approximateEpoch = self.nEvals // self.nParticles
            if approximateEpoch > self.nEpochs: break
            
            delayedParticle = delayed ( evaluate_particle ) ( self.scatteredDataFutures,
                                                              self.particles[pID].pos,
                                                              self.paramRanges,
                                                              self.particles[pID].pID,
                                                              self.dataset.trainObjective )
            
            futureParticle = self.client.compute( delayedParticle )
            particleFutureSeq.add( futureParticle )
            
# xgboost parameters -- https://xgboost.readthedocs.io/en/latest/parameter.html
def evaluate_particle ( scatteredDataFutures, particleParams, paramRanges, particleID, trainObjective, earlyStoppingRounds = 25 ) :
    trainDataFuture = scatteredDataFutures[0]
    trainLabelsFuture = scatteredDataFutures[1]
    testDataFuture = scatteredDataFutures[2]
    testLabelsFuture = scatteredDataFutures[3]
        
    xgboostParams = {
        'tree_method': 'gpu_hist',
        'random_state': 0, 
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
    xgboostParams['max_depth'] = enforce_type( particleParams[0], paramRanges[0] )
    xgboostParams['learning_rate'] = enforce_type( particleParams[1], paramRanges[1] ) # shrinkage of feature weights after each boosting step
    xgboostParams['gamma'] = enforce_type( particleParams[2], paramRanges[2] ) # complexity control, range [0, Inf ]
    xgboostParams['num_boost_rounds'] = 2000
    
    startTime = time.time()

    trainDMatrix = xgboost.DMatrix( data = trainDataFuture, label = trainLabelsFuture )
    testDMatrix = xgboost.DMatrix( data = testDataFuture, label = testLabelsFuture )

    trainedModelGPU = xgboost.train( dtrain = trainDMatrix, evals = [(testDMatrix, 'test')], params = xgboostParams,
                                     num_boost_round = xgboostParams['num_boost_rounds'], 
                                     early_stopping_rounds = earlyStoppingRounds,
                                     verbose_eval = False )

    trainDataPerf = 1 - float( trainedModelGPU.eval(trainDMatrix).split(':')[1] )
    testDataPerf = 1 - float( trainedModelGPU.eval(testDMatrix).split(':')[1] )   

    elapsedTime = time.time() - startTime

    return testDataPerf, trainDataPerf, particleID, elapsedTime