<img width = 85% src='rapids_motivation.png'>

<img width = 75% src='choices.png'>

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