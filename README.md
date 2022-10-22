## TorchGRL

Our new repository has been developed based on this old repository with a number of improvements, 
please go to the links of our new repository [Graph_CAVs](https://github.com/Jacklinkk/Graph_CAVs) for more information.
Subsequent improvements and development work will be carried out in the new repository.

TorchGRL is the source code for our paper **Graph 
Convolution-Based 
Deep Reinforcement Learning for Multi-Agent 
Decision-Making in Mixed Traffic Environments**.
TorchGRL is a modular simulation framework that 
integrates different GRL algorithms and SUMO 
simulation platform to realize the simulation 
of multi-agents decision-making algorithms 
in mixed traffic environment. 
You can  adjust the test scenarios and 
the implemented GRL algorithm according to your needs.

-------------------------------------
* [Preparation](#preparation)
* [Installation](#installation)
* [Instruction](#instruction)
* [Tutorial](#tutorial)


## Preparation
Before starting to carry out some relevant 
works on our framework, 
some preparations are required to be done.

### Hardware
Our framework is developed based on a laptop, and the specific configuration is as follows:
- Operating system: Ubuntu 20.04
- RAM: 32 GB
- CPU: Intel (R) Core (TM) i9-10980HK CPU @ 2.40GHz
- GPU: RTX 2070

It should be noted that our program must be reproduced under the Ubuntu 20.04 operating system, and we strongly recommend using GPU for training.

### Development Environment
Before compiling the code of our framework,
you need to install the following 
development environment:
- Ubuntu 20.04 with latest GPU driver
- Pycharm
- Anaconda
- CUDA 11.1
- cudnn-11.1, 8.0.5.39

## Installation
Please download our GRL framework 
repository first:
```
git clone https://github.com/Jacklinkk/TorchGRL.git
```

Then enter the root directory of TorchGRL:
```
cd TorchGRL
```

and **please be sure to run the 
below commands from /path/to/TorchGRL.**

### Installation of FLOW
The [FLOW](https://flow-project.github.io/usingFlow.html)
library will be firstly installed.

Firstly, enter the flow directory:
```
cd flow
```

Then, create a conda environment from flow library:
```
conda env create -f environment.yml
```

Activate conda environment:
```
conda activate TorchGRL
```

Install flow from source code:
```
python setup.py develop
```

### Installation of SUMO
[SUMO](https://www.eclipse.org/sumo/) simulation platform will be installed. 
**Please make sure to run the below commands 
in the "TorchGRL" virtual environment.**

Install via pip:
```
pip install eclipse-sumo
```

Setting in Pycharm:

In order to adopt SUMO correctly, 
you need to define the environment variable 
of SUMO_HOME in Pycharm. 
The specific directory is:
```
/home/…/.conda/envs/TorchGCQ/lib/python3.7/site-packages/sumo
```

Setting in Ubuntu:

At first, run:
```
gedit ~/.bashrc
```

then copy the path name 
of SUMO_HOME to “~/.bashrc”:
```
export SUMO_HOME=“/home/…/.conda/envs/TorchGCQ/lib/python3.7/site-packages/sumo”
```

Finally, run:
```
source ~/.bashrc
```

### Installation of Pytorch and related libraries
**Please make sure to run the below commands 
in the "TorchGRL" virtual environment.**

Installation of [Pytorch](https://pytorch.org/):

We use Pytorch version 1.9.0 for development
under a specific version of CUDA and cudnn.
```
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Installation of [pytorch geometric](https://pytorch-geometric.readthedocs.io/en/latest/#):

Pytorch geometric is a Graph Neural Network (GNN) library upon Pytorch
```
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
```

### Installation of pfrl library
**Please make sure to run the below commands 
in the "TorchGRL" virtual environment.**

[pfrl](https://github.com/pfnet/pfrl) 
is a deep reinforcement learning library 
that implements various algorithms 
in Python using PyTorch.

Firstly, enter the pfrl directory:
```
cd pfrl
```

Then install from source code:
```
python setup.py develop
```

## Instruction
### flow folder

The flow folder is the root directory of 
the library after the 
FLOW library is installed through 
source code, including interface-related 
programs between DRL algorithms and SUMO platform.

### Flow_Test folder 

The Flow_Test folder includes the related programs
of the test environment configuration; 
specifically, T_01.py is the core python program. 
If the program runs successfully, 
the environment configuration is successful.

### pfrl folder 
The pfrl folder is the root directory 
of the library after the 
deep reinforcement learning pfrl 
library is installed through 
source code, including all DRL related programs. 
The source program can be modified as needed.

### GRLNet folder 
The GRLNet folder contains the GRL neural network 
built in the Pytorch environment. 
You can modify the source code as needed 
or add your own neural network.

- Pytorch_GRL.py constructs the fundamental 
neural network of GRL algorithms
- Pytorch_GRL_Dueling.py constructs 
the dueling network of GRL algorithms

### GRL_utils folder 
The GRL_utils folder contains basic functions 
such as model training and testing, 
data storage, and curve drawing.
- Train_and_Test.py contains the 
training and testing functions 
for the GRL model.
- Data_Plot_Train.py is the function 
to plot the training data curve.
- Data_Process_Test.py is the function 
to process the test data.
- Fig folder stores the training data curve.
- Logging_Training folder stores the 
training data generated by different GRL algorithms.
- Logging_Test folder stores the testing data
generated by different GRL algorithms.

### GRL_Simulation folder 
The GRL_Simulation folder is the core 
of our framework, which contains the 
core simulation program and 
some related functional programs.

- main.py is the main program, 
containing the definition of FLOW parameters,
as well as the controlling (start and end) 
of the simulation.
- controller.py is the definition of 
vehicle control model based on FLOW library.
- environment.py is the core program to 
build and initialize the simulation 
environment of SUMO.
- network.py defines the road network.
- registry_custom.py registers the simulation 
environment of SUMO to the gym library 
to realize the connection with GRL algorithms.
- specific_environment.py defines the
elements in MDPs, including state representation, 
action space and reward function.
- Experiment folder is the core program of 
co-simulation under different GRL algorithms, 
including the initialization of the 
simulation environment, the initialization of 
the neural network, the training and testing 
of GRL algorithms, and the preservation of 
the training and testing results.
- GRL_Trained_Models folder stores the trained 
GRL model when the training process ends.

## Tutorial
You can simply run "main.py" in Pycharm to 
simulate the GRL algorithm, and observe the 
simulation process in SUMO platform. You can
generate training plot such as Reward curve:

### Verification of other algorithms
If you want to verify other algorithms, 
you can develop the source code as needed 
under the "Experiment folder", and don't 
forget to change the imported python script 
in "main.py". In addition, you can also construct 
your own network in GRLNet folder.

### Verification of other traffic scenario
If you want to verify other traffic scenario, 
you can define a new scenario in "network.py". 
You can refer to the documentation of 
SUMO for more details .

