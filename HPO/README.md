<img width = 85% src='rapids_motivation.png'>

<!-- <img width = 75% src='choices.png'> -->

# Hyperparamter Optimzation using RAPIDS and DASK

## Contents

1 -- synthetically generate a learning problem [ classification ]

2 -- go through a data science pipeline [ pre-processing , splitting , viz ]

3 -- model building [ xgboost ]
> hyper-parameters [ max-depth, nTrees, learning rate, regularization ... ]

> demonstrate performance [ CPU vs 1 GPU ]

4 -- scaling and hyper-parameter search
> dask + rapids [ xgboost ]
    
5 -- visualize search and reveal best model parameters

TODO - generate figures that capture benefit of GPU scaling

6 -- extensions [ multi-node [ dask kubernetes ], dask_xgboost [ larger dataset ] ]

## Install and Run Demo

1 -- clone repository

```git clone https://github.com/miroenev/rapids && cd rapids/HPO```

2 -- build container [takes 5-10 minutes]

```sudo docker build -t rapids-dask-hpo .```

3 -- launch/run the container [auto starts jupyterlab server]

```sudo rocker run --runtime=nvidia -it -p 8888:8888 -p 8787:8787 -p 8786:87876 rapids-dask-hpo```

4 -- connect to the notebook

> navigate browser to the IP of the machine running the container
e.g., http://127.0.0.1:8888

> In rapids/HPO open the rapids_dask_hpo.ipynb

## Using the CLI

1 -- get help on CLI options

``` python main.py --help```

2 -- launch experiments

```python main.py --num_gpus 4 --num_timesteps 10 --coil_type 'helix'```

## Using Kubeflow and Dask-Kubernetes

1 -- install Kubernetes, Ceph, and Kubeflow from [NVIDIA/deepops](https://github.com/NVIDIA/deepops/blob/master/docs/kubernetes-cluster.md)

2 -- navigate to the Kubeflow dashboard, select notebooks and create a new notebook server using the custom container

```ericharper/rapids-dask-hpo:latest```

3 -- change the run command

>\["-c", "/opt/conda/envs/rapids/bin/jupyter lab  --notebook-dir=/ --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=\$\{NB_PREFIX\}"\]

4 -- launch experiments via the notebook or the command line
```
!python main.py --k8s --adapt --num_gpus 4 --min_gpus 1 \
                --num_timesteps 10 --coil_type 'helix'
```
