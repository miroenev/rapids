import rapids_lib_v10 as rl

import dask
from dask import delayed
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

import argparse

def launch_dask(n_gpus):
    cluster = LocalCUDACluster(ip="", n_workers=n_gpus)
    client = Client(cluster)
    print(f'Launching Dask cluster with {n_gpus} GPUs')
    print(client)
    return client


def parse_args():
    parser = argparse.ArgumentParser(description='Perform hyper-parameter optimization using Dask',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # data generation arguments
    parser.add_argument('--coil_type', default='helix', type=str,
                        help='the type of coil to generate the data')
    parser.add_argument('--num_blobs', default=1000, type=int,
                        help='the number of blobs generated on the GPU')
    parser.add_argument('--num_coordinates', default=400, type=int,
                        help='the number of starting locations of each blob')
    parser.add_argument('--sdev_scale', default=.3, type=float,
                        help='standard deviation of normals used to generate data')
    parser.add_argument('--noise_scale', default=.1, type=float,
                        help='additional noise')
    parser.add_argument('--coil_density', default=12.0, type=float,
                        help='how tight the coils are')
    
    # ETL arguments
    parser.add_argument('--train_test_overlap', default=.05,
                        help='percentage of train and test distribution that overlaps')
    
    # HPO arguments
    parser.add_argument('--num_gpus', default=1, type=int,
                        help='the number of gpus to use')
    parser.add_argument('--num_timesteps', default=10, type=int,
                        help='the number of timesteps to run HPO')
    parser.add_argument('--num_particles', default=32, type=int,
                        help='the number of particles in the swarm')
    
    args = parser.parse_args()
    
    return args


def main(args):
    
    client = launch_dask(args.num_gpus)

    # generate data on the GPU
    data, labels, t_gen = rl.gen_blob_coils( coilType=args.coil_type, shuffleFlag = False, 
                                             nBlobPoints = args.num_blobs,  
                                             nCoordinates = args.num_coordinates, 
                                             sdevScales = [args.sdev_scale, args.sdev_scale, args.sdev_scale], 
                                             noiseScale = args.noise_scale,
                                             coilDensity = args.coil_density,
                                             plotFlag = False)
    
    # split
    trainData_cDF, trainLabels_cDF, testData_cDF, testLabels_cDF, t_split = \
        rl.split_train_test_nfolds ( data, labels, trainTestOverlap = args.train_test_overlap)

    # apply standard scaling
    trainMeans, trainSTDevs, t_scaleTrain = rl.scale_dataframe_inplace ( trainData_cDF )
    _, _, t_scaleTest = rl.scale_dataframe_inplace ( testData_cDF, trainMeans, trainSTDevs )
    
    # launch HPO on Dask
    nParticles = args.num_particles
    
    # TODO: add ranges to argparser
    paramRanges = { 0: ['max_depth', 3, 15, 'int'],
                    1: ['learning_rate', .001, 1, 'float'],
                    2: ['lambda', 0, 1, 'float'] }
    
    accuracies, particles, velocities, particleSizes, particleColors, \
    bestParticleIndex, bestParamIndex, particleBoostingRounds, \
    trainingTimes, _, elapsedTime = rl.run_hpo(client,
                                               args.num_timesteps,
                                               nParticles,
                                               paramRanges,
                                               trainData_cDF,
                                               trainLabels_cDF,
                                               testData_cDF,
                                               testLabels_cDF,
                                               plotFlag=False)
    
    # TODO: Scaling up: DGX-1, DGX2, Scaling out: Slurm, K8s 
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)