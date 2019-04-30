## Video Walkthrough of Notebook: [Video Link](https://www.dropbox.com/s/1qkmsnynog45ox8/rapids_walkthrough_4_25.mp4?dl=0)

## 1 -- clone repository

    git clone https://github.com/miroenev/rapids && cd rapids


## 2 -- build container [ takes 5-10 minutes ]

    sudo docker build -t rapids-demo:v0 .


## 3 -- launch/run the container [ auto starts jupyter notebook ]

    sudo docker run --runtime=nvidia -it --rm -p 8888:8888 -p 8787:8787 rapids-demo:v0


## 4 -- create a port map / ssh-tunnel to the container [ opening ports for jupyter lab and the dask dashboard ]
 	
    ssh -N -f -L 8888:localhost:8888 -L 8787:localhost:8787 remote_user@remote_host

## 5 -- launch demo from browser [ on your local machine ]

    use a browser [ firefox is recommended ] and navigate to localhost:8888    
    navigate to the /rapids directory and click on rapids_ml_workflow_demo.ipynb 

## 6 [ optional ] -- launch dask demo

    use a browser [ firefox is recommended ] and navigate to localhost:8888    
    launch/click on rapids_ml_workflow_demo_dask_v0.ipynb 
    open a browser tab to localhost:8787 for the dask dashboard
