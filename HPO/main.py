import data_utils # load datasets (or generate data) on the gpu
import swarm # particle swarm implementation

import cudf

import dask
from dask import delayed
from dask_cuda import LocalCUDACluster
from dask_kubernetes import KubeCluster
from dask.distributed import Client

from time import sleep # sleep needed to give K8S time for manual scaling
import argparse


default_worker_spec_fname = '/worker_spec.yaml'
default_worker_spec = '''
# worker-spec.yml

kind: Pod
metadata:
  labels:
    foo: bar
spec:
  restartPolicy: Never
  containers:
  - image: ericharper/rapids-dask-hpo:latest
    imagePullPolicy: IfNotPresent
    args: [dask-worker,  --nthreads, '1', --no-bokeh, --memory-limit, 6GB, --no-bokeh, --death-timeout, '60']
    name: dask
    resources:
      limits:
        cpu: "2"
        memory: 6G
        nvidia.com/gpu: 1
      requests:
        cpu: "2"
        memory: 6G
        nvidia.com/gpu: 1
'''


def launch_dask(n_gpus, min_gpus, k8s, adapt, worker_spec):
    if k8s:
        if worker_spec is None:
            worker_spec = default_worker_spec_fname
            print(f'Creating a default K8S worker spec at {worker_spec}')
            with open(worker_spec, "w") as yaml_file:
                yaml_file.write(default_worker_spec)
                
        cluster = KubeCluster.from_yaml(worker_spec)
        if adapt:
            cluster.adapt(minimum=min_gpus, maximum=n_gpus)
            print(f'Launching Adaptive K8S Dask cluster with [{min_gpus}, {n_gpus}] workers')
        else:
            cluster.scale(n_gpus)
            print(f'Launching K8S Dask cluster with {n_gpus} workers')
        sleep(10)
    else:
        cluster = LocalCUDACluster(ip="", n_workers=n_gpus)
        print(f'Launching Local Dask cluster with {n_gpus} GPUs')

    client = Client(cluster)
    print(client)
    print(cluster)
    return client, cluster


def close_dask(cluster, k8s):
    if k8s:
        cluster.scale(0)
        print("Shutting down Dask Pods")


def parse_args():
    parser = argparse.ArgumentParser(description='Perform hyper-parameter optimization using Dask', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Data generation arguments
    parser.add_argument('--dataset', default='synthetic', metavar='', type=str,
                        help="dataset to use: 'synthetic', 'higgs', 'airline'")
    parser.add_argument('--num_rows', default='10000', metavar='', type=int,
                        help='number of rows when using dataset')
    parser.add_argument('--coil_type', default='helix', metavar='', type=str,
                        help='the type of coil to generate the data')
    parser.add_argument('--num_blobs', default=1000, metavar='', type=int,
                        help='the number of blobs generated on the GPU')
    parser.add_argument('--num_coordinates', default=400, metavar='', type=int,
                        help='the number of starting locations of each blob')
    parser.add_argument('--sdev_scale', default=.3, metavar='', type=float,
                        help='standard deviation of normals used to generate data')
    parser.add_argument('--noise_scale', default=.1, metavar='', type=float,
                        help='additional noise')
    parser.add_argument('--coil_density', default=12.0, metavar='', type=float,
                        help='how tight the coils are')
    
    # ETL arguments
    parser.add_argument('--train_test_overlap', default=.05, metavar='', 
                        help='percentage of train and test distribution that overlaps')
    
    # HPO arguments
    parser.add_argument('--num_epochs', default=10, metavar='', type=int,
                        help='the number of times to evaluate all particles')
    parser.add_argument('--num_particles', default=32, metavar='', type=int,
                        help='the number of particles in the swarm')
    parser.add_argument('--async', default=False, dest='async_flag', action='store_true',
                        help='use asynch')
    parser.add_argument('--random_search', default=False, dest='random_search', action='store_true',
                        help='use random search: particles update randomly')
    
    # Cluster arguments
    parser.add_argument('--k8s', default=False, dest='k8s', action='store_true',
                        help='use a KubeCluster instead of LocalCudaCluster')
    parser.add_argument('--adapt', default=False, dest='adapt', action='store_true',
                        help='use adaptive scaling of k8s workers [min_gpus, num_gpus]')
    parser.add_argument('--spec', default=None, metavar='', type=str,
                       help='the k8s worker_spec yaml file to use')
    
    # Scaling arguments
    parser.add_argument('--num_gpus', default=1, metavar='', type=int,
                        help='the number of workers deployed or maximum workers when using K8S adaptive; each worker gets 1 GPU')
    parser.add_argument('--min_gpus', default=32, metavar='', type=int,
                        help='the minimum number of workers when using adaptive scaling')

    
    args = parser.parse_args()
    
    return args


def main(args):
    
    client, cluster = launch_dask(args.num_gpus, args.min_gpus,
                                  args.k8s, args.adapt, args.spec)
    
    if args.dataset == 'synthetic':
        # generate data on the GPU
        data, labels, t_gen = data_utils.generate_dataset( coilType = 'helix', nSamples = args.num_rows)

        # split
        trainData_cDF, trainLabels_cDF, testData_cDF, testLabels_cDF, t_split = \
            data_utils.split_train_test_nfolds ( data, labels, trainTestOverlap = args.train_test_overlap)

        # apply standard scaling
        trainMeans, trainSTDevs, t_scaleTrain = data_utils.scale_dataframe_inplace ( trainData_cDF )
        _, _, t_scaleTest = data_utils.scale_dataframe_inplace ( testData_cDF, trainMeans, trainSTDevs )
        
    
    # load data into GPU
    elif args.dataset =="higgs":
        df = data_utils.prepare_higgs("data/", args.num_rows)
    elif args.dataset == "airline":
        df = data_utils.prepare_airline("data/", args.num_rows)

    # convert to cuda dataframe
    if args.dataset == "higgs" or args.dataset == "airline":
        trainData_cDF = cudf.DataFrame.from_pandas(df.X_train)
        testData_cDF = cudf.DataFrame.from_pandas(df.X_test)
        trainLabels_cDF = cudf.DataFrame.from_pandas(df.y_train.to_frame())
        testLabels_cDF = cudf.DataFrame.from_pandas(df.y_test.to_frame())
    
    # launch HPO on Dask
    nEpochs = args.num_epochs
    nParticles = args.num_particles
    allowAsync = args.async_flag
    randomSearch = args.random_search
    
    # TODO: add ranges to argparser
    paramRanges = { 0: ['max_depth', 3, 20, 'int'],
                    1: ['learning_rate', .001, 1, 'float'],
                    2: ['gamma', 0, 2, 'float'] }
    
    mode = {'allowAsyncUpdates': args.async_flag, 'randomSearch': args.random_search }
    
    particleHistory, globalBest, elapsedTime = swarm.run_hpo(client,
                                               mode,
                                               paramRanges,
                                               trainData_cDF,
                                               trainLabels_cDF,
                                               testData_cDF,
                                               testLabels_cDF,
                                               nParticles,
                                               nEpochs,
                                               wMomentum = .05, 
                                               wIndividual = .25,
                                               wBest = .45, 
                                               wExplore = 0,
                                               plotFlag=False)
    
    # Shut down K8S workers
    close_dask(cluster, args.k8s)
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
