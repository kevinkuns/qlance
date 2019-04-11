# pytickle
PyTickle is a wrapper for the Matlab [Optickle](https://github.com/Optickle/Optickle/tree/Optickle2) simulation.

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
