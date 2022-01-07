# ts_analysis
**Time Series Analysis**: an analysis stream developed for the Kuperberg NeuroCognition of Language Lab. This package aspires to become a comprehensive processing package for analyzing time series data in a python environment.
This package is currently under development; there's no stable version yet.

## Author
Feng Cheng ([@tedchengf](https://github.com/tedchengf))

## Installation
Since the package is still under development and a stable version is not avaliable, you will need to use pip local installation. First you will need to  clone the repo in a suitable directory:

```bash
git clone https://github.com/tedchengf/ts_analysis.git
```
This will create the **ts_analysis** directory. Then use the pip local installation to install the package:
```bash
pip install ./ts_analysis 
```
Alternatively, if you want to edit the package on the run, install with -e flag:
```bash
pip install -e ./ts_analysis
```

## Structural Overview
Currently the package is divided into 6 subpackages, each with a targeted domain. Although functions and classes in each of the subpackages are relatively independent, some interdependency is inevitable. For example, the <code>RSA</code> class in <code>ts_analysis.imaging.multivariate</code>  depends on the <code>RDM</code> class in <code>ts_analysis.dataframes.rdm</code>.  

#### analysis
The subpackage <code>ts_analysis.analysis</code> is targeted for data analysis. Some of its core functions support analysis such as **PCA**, **MDS**, and **Hierarchical Clustering** (with their associated evaluation and plotting functions).

#### dataframes
The subpackage <code>ts_analysis.dataframes</code> contains various classes for holding data. <code>ts_analysis.dataframes.rdm</code> contains specialized classes `RDM` and `tsRDM`  that are optimized for storing and acessing elements in the **Representational Dissimilarity Matrix (RDM)**. <code>ts_analysis.dataframes.dframe</code> contains general-purpose dataframes for storing and accessing data. While classes like `DFrame` and `Dim` allow users to define an one dimensional arrays as annotations for each dimension of a `numpy.ndarray`, the class `cond` provides a convinent dataframe for storing and accessing dimensions, especially in scenarios where the user need to get intersections or unions of multiple dimensions.

#### imaging

The subpackage `ts_analysis.imaging` contains classes and functions that apply various imaging analyses. `ts_analysis.imaging.univariate` contains univariate analyses such as **Linear Mixed-Effects Regressions (LMERs)** models and **Event-Related Potential **. `ts_analysis.imaging.multivariate` contains multivariate analyses such as **Representational Similarity Analysis (RSA)** and its time-series variant.

#### inference

The subpackage `ts_analysis.inference` contains classes and functions for statistical inferences. Currently, the only class it offers is `CPerm_diff` which codes the statistical inference of **Cluster Permutation**.

#### plot

The subpackage `ts_analysis.plot` contains wrapper functions for `matplotlib`. `tsplot` contains functions are specialized for plotting time-series data (e.g., results from imaging analyses); `matplot` contains functions for visualizing matrices; `miscplot` contains other miscellaneous plotting functions.

#### utilities

The subpackage `ts_analysis.utilities` contains support functions for some of the classes and functions in other subpackages, but also contains numerous helpful functions that can potentially be incoporated for various analyses. `func` holds various functions that can be used for transforming vectors and matrices (e.g., distance functions); `matop` holds functions for matrix operations that are optimized and parallelized on the CPU level; `utilities` contains some miscellaneous support functions.  

## Usage
Due to the deepth of the package, I recommend seperate import commands for different subpackages. For example, to use the dataframes:
```python
import ts_analysis.dataframes as df
```
and then proceed to use the RDM:
```python
mat = df.rdm.RDM([1,2,3,4], "test matrix")
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