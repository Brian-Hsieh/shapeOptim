# Shape optimization in confined flow

This serves as a code base for the paper *Shape optimization in confined flow*.

## Description

Shape optimization is conducted using the differentiable physics toolkit [phiflow](https://github.com/tum-pbs/PhiFlow).
Currently, only Reynolds number of 40, 100 and 190 are supported of which the pre-calculated data are stored in the `prestored_data/` directory. 

To deal with respective goals listed below, different **methods** are implemented.
1. Update methods of the shape: **RK2, Adam**
2. Barycenter constraint methods: **shifting, Lagrange multiplier (LM)**
3. Vortex evaluation method (for unsteady flow): **Dynamic scanning (DS)**

The `code/` directory contains three python scripts which are the main implementation of the optimization procedure. The supported Reynolds numbers and the methods of repective scripts are shown below:

| Script name      | Reynolds number | Update of shape | Barycenter constraint | Vortex evaluation |
| :---             |    :----:       | :---:           | :---:                 | :---:             |
| energy_imm       | 40, 100, 190    | RK2 / Adam      | shifting              | N/A               |
| energy_LM_imm    | 40, 100, 190    | RK2             | LM                    | N/A               |
| energy_scan      | 100, 190        | RK2             | shifting              | DS                |

## Getting Started

### Dependencies

The code is originally run on Windows but it should be working on Linux as well. Python version of 3.6 or above is required.

The main required packages or libraries are listed below:
* phiflow 2.0.3 (see installation details below)
* torch 1.9.0 or 1.10.0
* scikit_fmm
* scipy
* numpy
* matplotlib
* varname

### Installing

To install [phiflow](https://github.com/tum-pbs/PhiFlow) with source code, run
```
pip install phiflow/
```

To install all required packages other than [phiflow](https://github.com/tum-pbs/PhiFlow) and Pytorch, run
```
pip install -r requirements.txt
```
To install Pytorch, please refer to the [Pytorch webpage](https://pytorch.org/get-started/locally/).

Since GPU acceleration is activated in default, CUDA installation is required. The CUDA installation recommendations can be found in [phiflow installation details](https://tum-pbs.github.io/PhiFlow/Installation_Instructions.html).

If GPU acceleration is not applicable, please comment out the following code in the scripts.
```python
TORCH.set_default_device('GPU')
```

### Executing program

Go to `code/` directory and run the script as usuall, e.g.:
```
cd code/
python energy_scan.py
```

Different arguments either required or optional ones can be specified in the command line. Use `python script.py -h` for details of the arguments.

The result data will be stored in the `data/` directory with the name `simXXX` specified by the user.

## Author

[Brian Hsieh](https://www.linkedin.com/in/meng-ju-hsieh-83a188162)