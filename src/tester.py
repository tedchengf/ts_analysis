# tester.py

# for ts_analysis.dataframes.dframe
import numpy as np
import ts_analysis.dataframes.dframe as df
import pickle

print("Testing Dim")
# testing Dim
identifier = [1,2,3,4,5,6,7,8,9,10]
test_dim = df.Dim("test", identifier)
print(test_dim)
print(np.array_equal([1,2,4,5], test_dim.slice([1,2,4,5])[1].identifier))
print(np.array_equal([4,5,2,1], test_dim.slice([4,5,2,1])[1].identifier))
print(np.array_equal([1,2,4,5], test_dim.slice([0,1,3,4], ktype = "index")[1].identifier))
print(np.array_equal([1,2,4,5], test_dim.slice([True,True,False,True,True,False,False,False,False,False], ktype = "index")[1].identifier))
print(np.array_equal([4,5,2,1], test_dim.slice([4,5,2,1])[1].identifier))
print(np.array_equal([1,2,4,5,6,8,9], test_dim.exclude([3,7,10])[1].identifier))
print(np.array_equal([1,2,4,5,6,8,9], test_dim.exclude([10,7,3])[1].identifier))
print(np.array_equal([1,2,4,5,6,8,9], test_dim.exclude([2,6,9], ktype = "index")[1].identifier))
test_dim.redefine(name = "new_test")
print(test_dim)
test_dim.redefine(identifier = [10,9,8,7,6,5,4,3,2,1])
print(test_dim.identifier)
try:
	test_dim.redefine(identifier = [1,1,1,1,1,1,1,1,1])
except RuntimeError:
	print(True)
try:
	test_dim.redefine(identifier = [1,1,1,1,1,1,1,1,1,1])
except KeyError:
	print(True)
print(test_dim == 5)
print(test_dim >= 5)
print(test_dim > 5)
print(test_dim <= 5)
print(test_dim < 5)
for identifier in test_dim: print(identifier)
test_dim.slice(slice(None, -1, None), ktype = "index")

print("Testing WrapperDim")
# testing WrapperDim
identifier = [1,2,3,4,5,6,7,8,9,10]
data_obj = [1,2,3,4,5,6,7,8,9,10]
test_wrapperdim = df.WrapperDim(data_obj, "test", identifier)
# test_wrapperdim = test_dim.wrap(data_obj)
print(test_wrapperdim)
print(test_wrapperdim[[1,2,5]])
print(np.array_equal([1,2,5], test_wrapperdim[[1,2,5]]))
print(np.array_equal([1,2,5], test_wrapperdim.slice([0,1,4], ktype = "index")))
print(test_wrapperdim.slice([1,2,5], return_self = True))
print(np.array_equal([1,2,3,4,5,6,7,9,10], test_wrapperdim.exclude(8)))
print(np.array_equal([1,2,3,4,5,6,7,9,10], test_wrapperdim.exclude(7, ktype = "index")))
test_wrapperdim.redefine(name = "new_test")
print(test_wrapperdim)

print("Testing DFrame")
data = np.arange(30).reshape((5,2,3))
test_df = df.DFrame(data, ["dim1", "dim2", "dim3"], [["one", "two", "three", "four", "five"], ["up","down"], [0,1,2]])
print(test_df)
f_data = pickle.dumps(test_df) 
test_df = pickle.loads(f_data)

test_wrapper = df.WrapperDim(test_df, "dim2", ["u","p"])
try:
	test_wrapper = df.WrapperDim(test_df, "dim3", ["u","p"])
	print("Dim Match Problem")
except:
	print("True")
print(test_wrapper[["p", "u"]].dim2.identifier)  

print(np.array_equal(test_df.dim1["one"].data, test_df.data[0]))
print(np.array_equal(test_df.dim1[["one", "three"]].data, test_df.data[[0,2]]))
print(np.array_equal(test_df[:2].data, test_df.data[:2]))
print(np.array_equal(test_df[:,:,2].data, test_df.data[:,:,2]))
print(np.array_equal(test_df[1,:,2].data, test_df.data[1,:,2]))
print(np.array_equal(test_df[1:4,:,:2].data, test_df.data[1:4,:,:2]))
print(np.array_equal(test_df[1,0,2], test_df.data[1,0,2]))
print(np.array_equal(test_df.slice(["one", "three"], "dim1").data, test_df.data[[0,2]]))
print(np.array_equal(test_df.slice([0, 2], "dim1", ktype = "index").data, test_df.data[[0,2]]))
print(np.array_equal(test_df.slice([True, False, True, False, False], "dim1", ktype = "index").data, test_df.data[[0,2]]))
print(np.array_equal(test_df.dim1.slice([True, False, True, False, False], ktype = "index").data, test_df.data[[0,2]]))
# print(test_df.slice([True, False, True, False, False], "dim1", ktype = "index").data)
# print(test_df.dim1[["one", "three"]].data)
print(np.array_equal(test_df.exclude(["two", "four", "five"], "dim1", ktype = "identity").data, test_df.data[[0,2]]))
print(test_df.dim1["one"].dim3 < 2)
