import data_utils # load datasets (or generate data) on the gpu
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
    logFilename = 'local_cluster_experiment_results.csv'
    if not os.path.isfile( logFilename ):
        with open( logFilename, mode='w+') as outputCSV:
            outputCSV.write("elapsedTime,nSamples,asyncMode,nGPUs,nParticles,nEpochs,globalBestAccuracy,globalBest_max_depth,globalBest_learning_rate,globalBest_gamma,globalBest_nTrees,datasetName\n")
    
    for iDataSamples in args.num_rows:
        # generate or load data directly to the GPU
        if args.dataset == 'synthetic':
            data, labels, t_gen = data_utils.generate_dataset( coilType = 'helix', nSamples = iDataSamples)
        elif args.dataset =="higgs":
            data, labels, t_gen = data_utils.load_higgs_dataset("data/higgs", iDataSamples)
        elif args.dataset == "airline":
            data, labels, t_gen = data_utils.load_airline_dataset("data/airline", iDataSamples)
        elif args.dataset == "fashion-mnist":
            data, labels, t_gen = data_utils.load_fashion_mnist_dataset("data/fmnist", iDataSamples)

        # split data into train and test
        if args.dataset == 'synthetic':
            trainData, trainLabels, testData, testLabels, _ = data_utils.split_train_test_nfolds ( data, labels, trainTestOverlap = args.train_test_overlap )
        else:
            trainData, testData, trainLabels, testLabels = cuml.train_test_split( data, labels, shuffle = False, train_size=args.train_size )

        # apply standard scaling
        trainMeans, trainSTDevs, t_scaleTrain = data_utils.scale_dataframe_inplace ( trainData )
        _, _, t_scaleTest = data_utils.scale_dataframe_inplace ( testData, trainMeans, trainSTDevs )    

        print ( f'training data shape : {trainData.shape}, test data shape {testData.shape}' )
        
        for iGPUs in args.num_gpus:
            for iParticles in args.num_particles:
                for iEpochs in args.num_epochs:
                    
                    mode = {'allowAsyncUpdates': args.async_flag, 'randomSearch': False }
                            
                    client, cluster = launch_dask(iGPUs)

                    particleHistory, globalBest, elapsedTime = swarm.run_hpo ( client, mode, paramRanges, 
                                                                                   trainData, trainLabels, testData, testLabels,
                                                                                   iParticles, iEpochs )
                                        
                    stringToOutput = f"{elapsedTime},{iDataSamples},{args.async_flag},{iGPUs},{iParticles},{iEpochs},"
                    stringToOutput += f"{globalBest['accuracy']},{globalBest['params'][0]},{globalBest['params'][1]},{globalBest['params'][2]},"
                    stringToOutput += f"{globalBest['nTrees']},{args.dataset}\n"
                    print( stringToOutput )
                    
                    with open(logFilename, mode='a') as outputCSV:                        
                        outputCSV.write(stringToOutput)
                    
                    print( 'closing dask cluster in between experiment runs [ sleeping for 5 seconds ]')
                    
                    client.close()
                    cluster.close()                    
                    time.sleep(5)                    
    
if __name__ == '__main__':
    args = parse_args()
    experiment_harness ( args )