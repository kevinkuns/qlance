# PyTickle

PyTickle is a python package that simulates optomechanical systems using the Optickle and/or Finesse packages. It adds functionality to these programs making them more user friendly and making it easier to compare the results. PyTickle also models control loops involving optomechanical plants defined by either Optickle or Finesse and can find state space representations of these plants.

* [Optickle](https://github.com/Optickle/Optickle/tree/Optickle2) is a Matlab-based optomechanical simulaton package. In order to use these simulations, PyTickle thus requires Matlab to be installed; however, the user only ever needs to use python while Matlab runs in the background. Matlab only needs to be installed to run the Optickle simulations and is not needed for the Finesse simulations or the control loop calculations.
* [Finesse](http://www.gwoptics.org/finesse/) is a C-based optomechanical simulaton package that can also be run entirely in python.
* Parts of the architecture of PyTickle's native control loop simulations are inspired by a [study of the advanced LIGO angular sensing and control system](https://iopscience.iop.org/article/10.1088/0264-9381/27/8/084026) and [lentickle](https://github.com/nicolassmith/lentickle), another Matlab package which is not needed to run PyTickle.
* State space representations of the optomechanical plants are found through an interface with [IIRrational](https://lee-mcculler.docs.ligo.org/iirrational/) and can be converted to Matlab-like systems from the python [control systems](https://python-control.readthedocs.io/en/0.8.3/index.html) package which allow for more sophisticated design techniques.

## Getting started
  1. Clone Optickle2 from [this](https://github.com/Optickle/Optickle/tree/Optickle2) repository. It is important to get Optickle2 and not Optickle.
  1. Install the python Matlab engine as described [here](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
  1. Clone this repository
  1. Optionally, set the variable `OPTICKLE_PATH` to the path to the Optickle2 directory.
  
## Basic usage

Before using PyTickle, the python matlab engine needs to be initialized. This takes some time and only needs to be done once per session
```python
>>> import matlab.engine
>>> eng = matlab.engine.start_matlab()
```
The path to Optickle2 then needs to be added to Matlab's path:
```python
>>> import pytickle as pyt
>>> pyt.addOpticklePath(eng, optickle_path)
```
If the variable `OPTICKLE_PATH` is defined, you can also just use
```python
>>> pyt.addOpticklePath(eng)
```

## Control Systems

![Control Loop](documentation/control_loop.svg)
