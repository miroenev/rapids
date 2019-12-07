import data_utils # load datasets (or generate data) on the gpu
from data_utils import Dataset
import swarm # particle swarm implementation

import cudf
import cuml

import time
import os

import dask
from dask import delayed
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

import argparse
# run_local_experiments --dataset 'synthetic' --async True  --num_rows 100000 1000000 10000000 --num_gpus 1 4 8 16 --num_epochs 10 --num_particles 32
# run_local_experiments --dataset 'synthetic' --async False --num_rows 100000 1000000 10000000 --num_gpus 1 4 8 16 --num_epochs 10 --num_particles 32

def parse_args():
    parser = argparse.ArgumentParser( description = 'Perform hyper-parameter optimization using Dask',
                                      formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    
    # Data generation arguments
    parser.add_argument('--dataset', default='synthetic', metavar='', type=str,
                        help="dataset to use: 'synthetic', 'higgs', 'airline', 'fashion-mnist")

    parser.add_argument('--num_rows', nargs='+', default='10000', metavar='', type=int,
                        help='number of rows when using dataset')
    
    # ETL arguments
    parser.add_argument('--train_test_overlap', default=.025, metavar='', 
                        help='percentage of train and test distribution that overlaps for synthetic data')
    
    parser.add_argument('--train_size', default=.7, metavar='', type=float,
                        help='percentage of data used for training with real datasets')
    
    # HPO arguments
    parser.add_argument('--num_epochs', nargs='+', default=10, metavar='', type=int,
                        help='the number of times to evaluate all particles')
    
    parser.add_argument('--num_particles', nargs='+', default=32, metavar='', type=int,
                        help='the number of particles in the swarm')
    
    parser.add_argument('--async', default=False, dest='async_flag', action='store_true',
                        help='use async')
    
    # Scaling arguments
    parser.add_argument('--num_gpus', nargs='+', default=1, metavar='', type=int,
                        help='the number of workers deployed or maximum workers when using K8S adaptive; each worker gets 1 GPU')
    
    args = parser.parse_args()    
    return args

def launch_dask(n_gpus):
    cluster = LocalCUDACluster(ip="", n_workers=n_gpus)
    print(f'Launching Local Dask cluster with {n_gpus} GPUs')

    client = Client(cluster)
    print(client)
    print(cluster)
    return client, cluster

def experiment_harness ( args ):

    experimentLog = {}
    experimentCount = 0 

     # TODO: add ranges to argparser
    paramRanges = { 0: ['max_depth', 3, 20, 'int'],
                    1: ['learning_rate', .001, 1, 'float'],
                    2: ['gamma', 0, 2, 'float'] }   
    
    client = cluster = None
    
    # create logfile and write headers
    logFilename = 'results.csv'
    if not os.path.isfile( logFilename ):
        with open( logFilename, mode='w+') as outputCSV:
            outputCSV.write("elapsedTime,nSamples,asyncMode,nGPUs,nParticles,nEpochs,globalBestAccuracy,globalBest_max_depth,globalBest_learning_rate,globalBest_gamma,globalBest_nTrees,datasetName\n")
    
    for iDataSamples in args.num_rows:
        # generate or load data directly to the GPU
        if args.dataset == 'synthetic':
            dataset = Dataset('synthetic', iDataSamples)
        if args.dataset == 'airline':
            dataset = Dataset('airline', iDataSamples)
        if args.dataset == 'fashion-mnist':
            dataset = Dataset('fashion-mnist', iDataSamples)

        for iGPUs in args.num_gpus:
            for iParticles in args.num_particles:
                for iEpochs in args.num_epochs:
                    client, cluster = launch_dask(iGPUs)
                    if args.async_flag:
                        s = swarm.AsyncSwarm(client, dataset, paramRanges=paramRanges,
                                            nParticles=iParticles, nEpochs=iEpochs)
                    else:
                        s = swarm.SyncSwarm(client, dataset, paramRanges=paramRanges,
                                            nParticles=iParticles, nEpochs=iEpochs)
                    startTime = time.time()
                    s.run_search()
                    elapsedTime = time.time() - startTime

                    # TODO: remove fake nTrees
                    s.globalBest['nTrees'] = 9999

                    stringToOutput = f"{elapsedTime},{iDataSamples},{args.async_flag},{iGPUs},{iParticles},{iEpochs},"
                    stringToOutput += f"{s.globalBest['accuracy']},{s.globalBest['params'][0]},{s.globalBest['params'][1]},{s.globalBest['params'][2]},"
                    stringToOutput += f"{s.globalBest['nTrees']},{args.dataset}\n"
                    print( stringToOutput )
                    
                    with open(logFilename, mode='a') as outputCSV:                        
                        outputCSV.write(stringToOutput)
                    
                    print( 'closing dask cluster in between experiment runs [ sleeping for 5 seconds ]')
                    
                    client.close()
                    cluster.close()                    
                    time.sleep(5)                    
    
if __name__ == '__main__':
    args = parse_args()
    experiment_harness( args )