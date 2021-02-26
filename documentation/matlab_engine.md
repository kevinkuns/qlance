# Installing the Python MATLAB Engine

The Python MATLAB engine must be installed to run the Optickle simulations. This is not necessary if you are not interested in running Optickle.

## Using system python
To install the python MATLAB engine using the system python, follow the instructions from MATLAB [here](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).

## Using a conda environment
To install the python MATLAB engine in a conda environment, you will probably have to follow the instructions for installing in a non-default location [here](https://www.mathworks.com/help/matlab/matlab_external/install-matlab-engine-api-for-python-in-nondefault-locations.html).
For the base conda environment, the proper path is probably something like
```console
cd "matlabroot/extern/engines/python"
python setup.py install --prefix="~/anaconda3"
```
(To find the `matlabroot` path, open a matlab terminal and enter `matlabroot`.)
Or, to install in an environment named `qlance`, for example, the path is probably something like
```console
cd "matlabroot/extern/engines/python"
python setup.py install --prefix="~/anaconda3/envs/qlance"
```

To double check the correct paths, open a python terminal and enter
```python
>>> import sys
>>> sys.path
```
In the first case of the base environment, `~/anaconda3/lib/python3.7/site-packages` should be one of the entries.
In the second case of an environment named `qlance`, `~/anaconda3/envs/qlance/lib/python3.7/site-packages` should be one of the entries.
If the paths are slightly different, change the paths specified by `prefix` by analogy.
