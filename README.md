# ts_analysis
Time Series Analysis: an analysis stream created for the Kuperberg NeuroCognition of Language Lab.
This package is currently under development

## Author
Feng Cheng ([@tedchengf](https://github.com/tedchengf))

## Installation
Since the package is still under development and a stable version is not avaliable, you will need to use pip local installation. First you will need to  clone the repo in a suitable directory:
```bash
git clone https://github.com/tedchengf/ts_analysis.git
```
This will create the **ts_analysis** directory. Then use the pip local installation to pip install the package:
```bash
pip install ./ts_analysis 
```
Alternatively, if you want to edit the package on the go, install with -e flag:
```bash
pip install -e ./ts_analysis
```

## Usage
Currently the RSA related function are stored in ts_analysis.imaging_analysis. For example, to use the functions in the rdm file:
```python
from ts_analysis.imaging_analysis import rdm
```
and then proceed to use its functions/classes:
```python
mat = rdm.RDM([1,2,3,4], "test matrix")
```
## Requirements
* numpy
* numba
* scipy
* pandas
* gensim
* matplotlib
* tqdm
* sklearn
