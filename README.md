# Gail Navigation

This is a GAIL[1] based local navigation planner. 

## Requriements
- ROS2 Foxy
- CUDA 12.3 
- Python 3.8.10
    - Other python package versions are given in the requirements.txt

## Installation
There are three packagaes inside this folder which has to be installed.
Therefore it is recommended to create a python virtual enviroment(has less
issues with ros2 than conda virtual enviroments)
- GailNavigationNetwork pytorch model
- kris_envs gymansium enviroment
- gail_navigation ros2 package

Installation instructions are given in each package README file 

## References 
[1] J. Ho and S. Ermon, “Generative adversarial imitation learning,” in Advances in Neural Infor-
mation Processing Systems, pp. 4565–4573, 2016.
