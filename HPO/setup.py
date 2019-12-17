from distutils.core import setup

# Needed to enable importing for the HPO library for K8s-dask-workers 
setup( name = 'rapids-dask-hpo',
       version='0.0', description='Particle-Swarm HPO Library Package', 
       py_modules=['swarm', 'data_utils', 'visualization'])
