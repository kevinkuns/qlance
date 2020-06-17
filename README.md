# PyTickle

PyTickle is a python package that simulates optomechanical systems and quantum noise using the Optickle and/or Finesse packages. It adds functionality to these programs making them more user friendly and making it easier to compare the results. PyTickle also models control loops involving optomechanical plants defined by either Optickle or Finesse and can find state space and zpk representations of these plants.

* [Optickle](https://github.com/Optickle/Optickle/tree/Optickle2) is a MATLAB-based optomechanical simulaton package. In order to use these simulations, PyTickle thus requires Matlab to be installed; however, the user only ever needs to use python while Matlab runs in the background. MATLAB only needs to be installed to run the Optickle simulations and is not needed for the Finesse simulations or the control loop calculations.
* [Finesse](http://www.gwoptics.org/finesse/) is a C-based optomechanical simulaton package that can also be run entirely in python.
* Parts of the architecture of PyTickle's native control loop simulations are inspired by a [study of the advanced LIGO angular sensing and control system](https://iopscience.iop.org/article/10.1088/0264-9381/27/8/084026) and [lentickle](https://github.com/nicolassmith/lentickle), another MATLAB package which is not needed to run PyTickle.
* State space and zpk representations of the optomechanical plants are found through an interface with [IIRrational](https://lee-mcculler.docs.ligo.org/iirrational/).

## Setup
  1. Install PyTickle using pip by downloading the latest release from this repository and using
  ```shell
  pip install --user pytickle-version.tar.gz
  ```
  
  2. If you want to use Optickle simulations
     1. Clone Optickle2 from [this](https://git.ligo.org/IFOsim/Optickle2) repository.
     2. Install the python MATLAB engine as described [here](https://github.com/kevinkuns/pytickle/blob/master/documentation/matlab_engine.md)
     3. Optionally, set the variable `OPTICKLE_PATH` to the path to the Optickle2 directory.
    
  3. If you want to use Finesse simulations, install Finesse and pykat as described [here](https://git.ligo.org/finesse/pykat#installation).
  4. If you want to find state space representations of the optomechanical plants, install IIRrational as described [here](https://lee-mcculler.docs.ligo.org/iirrational/install.html#install).
  
## Getting Started
  
  1. See the [examples](examples) for basic usage.
     1. Start with the [Fabry Perot](examples/FabryPerot) examples for basic model building and analysis examples.
     2. Look at the [FPMI](examples/FPMI) examples for a detailed MIMO control loop example.
    
  2. See [here](documentation/optickle_vs_finesse.md) for an incomplete discussion of the differences between Optickle and Finesse.
  
  3. See [here](documentation/control_systems.md) for a reference of the PyTickle control loop topology.
