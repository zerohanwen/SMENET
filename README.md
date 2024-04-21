# MVDSNET

[![Powered by ](https://img.shields.io/badge/Powered%20by-***University%20-orange.svg?style=flat&colorA=555&colorB=-8A2BE2)]
**
## MVDSNET: A Multi-view Semantic Model for Multi-level Enzyme Function Prediction

Enzymes play a crucial role as ### catalysts. Comprehending biological ### and cellular ### is facilitated by the Enzyme Commission (EC), which matches protein ### to the biochemical reactions they catalyse through EC numbers.（A full introduction will be received in the paper as an update）
- **Source code**: https://github.com/zerohanwen/MVDSNET

## Installation

**MVDSNET** support Python 3.6+, Additionally, you will need
```pandas```, ```numpy```, ```scikit-learn```, ```torch```.
However, these packages should be installed automatically when installing this codebase.

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ecpick)](https://pypi.org/project/ecpick/)
              
![PyPI](https://img.shields.io/badge/scikitlearn-1.3.0-green)
![PyPI](https://img.shields.io/badge/numpy-1.24.1-green)
![PyPI](https://img.shields.io/badge/torch-2.0.1+cu117-green)
## Documentation

### embedding
The enzyme sequence is embedded using ESM-2：https://github.com/facebookresearch/esm
```shell
embedding
```
### train
```shell
train
```
### test
#### task1
```shell
### use task1 model get yes or no Enzyme
### output 1(enzyme) 2(noenzyme)
```
#### task2
```shell
### use task2 model get the EC number
### output 1,34,56,102.,...,231
### use final_EC_get/result_get_g.ipynb get the EC numbr: 1.1.1.1,2.3.3.1 ...
```
（A full Documentation will be received in the paper as an update）
