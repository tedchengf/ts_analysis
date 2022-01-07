# ts_analysis.dataframes

The subpackage <code>ts_analysis.dataframes</code> contains various data containers. <code>ts_analysis.dataframes.rdm</code> contains specialized classes `RDM` and `tsRDM`  that are optimized for storing and acessing elements in the **Representational Dissimilarity Matrix (RDM)**. <code>ts_analysis.dataframes.dframe</code> contains general-purpose dataframes for storing and accessing data. While classes like `DFrame`,  `Dim`, and  `WrapperDim` allow users to define an one dimensional arrays as annotations for each dimension of a `numpy.ndarray`, the class `cond` provides a convinent dataframe for storing and accessing dimensions, especially in scenarios where the user need to get intersections or unions of multiple dimensions.

## A brief introduction to dataframes (for now)

At the core of the idea of all dataframes is **identifier**. Normally, when you try to slice a numpy ndarray or a python list you use **indexes** to indicate the elements you want:

```python
>>> import numpy as np
>>> test_arr = np.array([0,1,2,3,4,5])
>>> test_arr[:4]
array([0, 1, 2, 3])
```

However, indexes themselves do not express any meaningful content about the sliced subarray. For multidimensional neural imaging data, accurately slicing the correct subarrays from the data demands meticulous attention and can be a great source of error in my past coding experiences. One way to simplify this operation is to use dictionaries, where each element is accessed by a unique **key**:

```python
>>> test_dict = dict({"a":1, "b":2, "c":3})
>>> test_dict["b"]
2
```

But it is a bit inconvinent to make a dictionary and a ndarray works together. This minor inconvinence motivates me to write several dataframes that allow the user to define **identifier** for their data. Much like **keys**, **identifiers** are unique labels a user can assign to define or annotate a dimension of an ndarray. For the rest of this introduction, please keep in mind that there are two modes of accessing the data:

1. **index mode** allows the user to slice the data by their **indexes**
2. **identity mode** allows the user to slice the data by a list of user-defined **identifiers**

For most of the dataframes offered here, the slicing operates by default on **identity mode**

#### ts_analysis.dataframes.dframe.WrapperDim

The `WrapperDim` class annotates the first dimension of an *iterable* object (in this case it's a list) with an list of identifiers:

```python
>>> from ts_analysis.dataframes import dframe as df
>>> ## the first param is the iterable object, the second is the name of the WrapperDim, and the third is the identifier
>>> test_wrapper = df.WrapperDim([10,9,8,7], "test", ["1","2","3","4"])
>>> test_wrapper[:]
array([10,  9,  8,  7])
>>> test_wrapper[["2","4"]]
array([9, 7])
>>> print(test_wrapper)                                                                   
"WrapperDim 'test' with 4 identifiers as <U1"
>>> test_wrapper.identifier
array(['1', '2', '3', '4'], dtype='<U1')
>>> test_wrapper.data_obj                                                                 
array([10,  9,  8,  7])
```

Note that by default the `WrapperDim` slice elements by the **identifiers**, not **indexes**. You can access the defined identifiers and data through the attributes `identifier` and `data_obj` respectively. There are also some other functions and overloaded default operations to make your life a bit easier:

```python
>>> test_wrapper == "4"
array([7])
>>> test_wrapper.slice([1,3], ktype = "index") 
array([9, 7])
>>> test_wrapper.exclude("1")                                                             
array([9, 8, 7])
```

Finally, `WrapperDim` is an *iterable* type. Importantly, unlike nomral lists or numpy arrays, the `WrapperDim` object behaves differently when you simply get the elements through the `__getitem__` function and when you iterate through the object through the `__iter__` function. While the former will give you a slice of its embedded `data_obj`, the latter will give you an iterator of its `identifier`:

```python
## Using the __getitem__ function overload                                                       
>>> test_wrapper["1"]
array([10])
## Using the __iter__ function overload                                               
>>> for i in test_wrapper:                                                                
...     print(i)
... 
1
2
3
4
```

This difference in behavior is admittedly unintuitive. But separating those behavior provide (imo) a great syntactic sugar for easy coding. For example, this allow you to quickly use the `WrapperDim` simply as an identifier when you are interacting with some other objects:

```python
>>> test_wrapper2 = df.WrapperDim([11,22,33,44,55,66], "test2", ["1","2","3","4","5","6"])  
>>> test_wrapper2[test_wrapper]                                                           
array([11, 22, 33, 44])
## Which is equivalent to the following: 
>>> test_wrapper2[["1","2","3","4"]]                                                      
array([11, 22, 33, 44])
```

This seperation should also appear a bit more intuitive when it is incoporated in `ts_analysis.dataframes`, which I will introduce now.

#### ts_analysis.dataframes.dframe.DFrame

The `DFrame` class annotes all dimensions of a *numpy ndarray*. It works like pandas, except it has much less functions but can handle more than two dimensons. It is also essentially an augmented `WrapperDim` class, where each dimension of ndarray corresponds to a `WrapperDim`. The `DFrame` class additionally handles these `WrapperDim` objects through a higher level **dim identifiers**, where each dimension of the ndarray will be given an *str* identifier to be refered to. This dim identifier will also be represented in the `name` attribute of the corresponding `WrapperDim` class. Here's a toy example:

```python
>>> data = np.arange(30).reshape(3,5,2)                                                   
>>> test_df = df.DFrame(data, ["trial", "timepoints", "channels"], [[1,2,3],[0, 10, 20, 30, 40],["Cz","Pz"]])
>>> print(test_df)
"DFrame Object with shape (3, 5, 2)"
"        - 0: WrapperDim 'trial' with 3 identifiers as int64"
"        - 1: WrapperDim 'timepoints' with 5 identifiers as int64"
"        - 2: WrapperDim 'channels' with 2 identifiers as <U2"
>>> print(test_df.trial)                                                                  
"WrapperDim 'trial' with 3 identifiers as int64"
```

Like pandas dfs, `DFrame` allows the user to dynamically define the class attribute. Once you defined the name of a dimension, you can directly access the `WrapperDim` of that dimension like accessing a class attribute. You can also change the name of the dimensions on the go, and it's corresponding attribute will be changed:

```python
>>> test_df.redefine_dim("trial", name = "trials")
>>> print(test_df)
"DFrame Object with shape (3, 5, 2)"
"        - 0: WrapperDim 'trials' with 3 identifiers as int64"
"        - 1: WrapperDim 'timepoints' with 5 identifiers as int64"
"        - 2: WrapperDim 'channels' with 2 identifiers as <U2"
>>> print(test_df.trials)                                                                 
"WrapperDim 'trials' with 3 identifiers as int64"
>>> print(test_df.trial)
"Traceback (most recent call last):"
"  File '<stdin>', line 1, in <module>"
"  File '/Users/feng/Desktop/ts_analysis/src/ts_analysis/dataframes/dframe.py', line 283, in __getattr__"
"    if name not in self.__dim_dict: raise AttributeError"
"AttributeError"
```

A bit more have to be said about slicing the elements in `DFrame`. Recall that `DFrame` access it's `WrapperDim` through a **dim identifier**. This means that whenever you want to slice `DFrame` on a particular dimension you need to provide both the **dim identifier** and the **identifiers** of this dimension. This is hard to accomadate through overloading the `__getitem__` function on `DFrame`. For this reason, the `__getitem__` function of `DFrame` operates on **index mode** much like a numpy array. To slice on **identity mode**, you will need to invoke the `WrapperDim` and use its `__getitem__` function. It all sounds complex, but it should be intuitive once you see how they operates:

```python
## index mode through the __getitem__ function on the DFrame object
>>> slice_1_df = test_df[:,:3,:]                                                          
>>> print(slice_1_df)                                                                     
"DFrame Object with shape (3, 3, 2)"
"        - 0: WrapperDim 'trials' with 3 identifiers as int64"
"        - 1: WrapperDim 'timepoints' with 3 identifiers as int64"
"        - 2: WrapperDim 'channels' with 2 identifiers as <U2"
## identity mode through the __getitem__ function implemented on the corresponding WrapperDim object
>>> slice_2_df = test_df.timepoints[[0, 10, 20]]                                          
>>> print(slice_2_df)
"DFrame Object with shape (3, 3, 2)"
"        - 0: WrapperDim 'trials' with 3 identifiers as int64"
"        - 1: WrapperDim 'timepoints' with 3 identifiers as int64"
"        - 2: WrapperDim 'channels' with 2 identifiers as <U2"
>>> print(np.array_equal(slice_1_df.data, slice_2_df.data))                               
True
## you can use other functions in WrapperDim for slicing
>>> slice_3_df = test_df.timepoints.exclude([30,40])
>>> print(np.array_equal(slice_2_df.data, slice_3_df.data))                               
True
## you can also make use of some overloaded math operations for quick slicing
>>> slice_4_df = test_df.timepoints < 30   
>>> print(np.array_equal(slice_2_df.data, slice_4_df.data)) 
True
```

 Finally, recall that `WrapperDim` has different behavior under `__getitem__` and `__iter__`. You can understand the `WrapperDim` embedded in a `DFrame` as a "slice" of the dimensions of `DFrame`. Intuitively, the user should be able to use the identifiers recorded on a particular dimension to be directly usable in other operations such as comparison. By seperating the behavior of  `__getitem__` and `__iter__` the user can do this:

```python
>>> np.array_equal(test_df.timepoints, [0,10,20,30,40])                                   
True
## Finding intersection of identifiers
>>> list(set(test_df.timepoints) & set([0,10,40,70]))                                     
[0, 10, 40]
## Using the identifier of one DFrame to slice another
>>> print(test_df.timepoints[slice_3_df.timepoints])                                      
"DFrame Object with shape (3, 3, 2)"
"        - 0: WrapperDim 'trials' with 3 identifiers as int64"
"        - 1: WrapperDim 'timepoints' with 3 identifiers as int64"
"        - 2: WrapperDim 'channels' with 2 identifiers as <U2"
```

Admittedly, this makes the `WrapperDim` a less well-defined and conceptually-clear class, only to reduce one dot operation. Also, in this manner the user cannot directly iterate through the content in a dimension of the ndarray without going back to the **index mode**. I might change this syntax to be more standard in the future, or not.

 
