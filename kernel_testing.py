
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import skcuda.cublas as cublas
import os

here = os.path.dirname(os.path.abspath(__file__))
here = "/home/dev/cuda-venv/src/quickzonoreach/"
kernel_filename = "verts_kernel.cu" 
kernel_filepath = os.path.join(here, kernel_filename)
with open(kernel_filepath, "r") as kernel_file:
    kernel_source_string = '\n'.join(kernel_file.readlines())
    module = SourceModule(kernel_source_string, options=['-g'])
    
    matmul_gpu = module.get_function("matmul_global")
    matvec_gpu = module.get_function("matvec_global")
    matadd_scalar_gpu = module.get_function("matadd_scalar_global")
    matmul_scalar_gpu = module.get_function("matmul_scalar_global")
    sumveclist_gpu = module.get_function("sum_vec_list_global")
    sum_vec_inplace_gpu = module.get_function("sum_vec_inplace_global")
    sum_veclist_gpu = module.get_function("sum_vec_list_global")

import numpy as np
#float*, float*, float*, int*
def matmul(A, B, C, dims):
    return np.matmul(A, B)

#float*, float*, float*, int*
def matvec(A, B, C, dims):
    return np.dot(A, B)

#float*, float, float*, int*
def matadd_scalar(A, B, C, dims):
    return A + B

#float*, float, float*, int*
def matmul_scalar(A, B, C, dims):
    return A * B

#float *, float *, int* 
def sum_vec_inplace(A, B, length):
    return A + B

#float*, int , int, int
def sumveclist(A, spacing, length, vec_size):
    return np.sum(A, axis=0)


def to_gpu(obj):
    obj_gpu = cuda.mem_alloc(obj.nbytes)
    cuda.memcpy_htod(obj_gpu, obj)
    return obj_gpu

def test_matmul(A, B):
    dims = np.asarray(list(A.shape) + list(B.shape), dtype=np.int32)
    C = np.zeros((dims[0],dims[3])).astype(np.float32)
    grid_dims = (1, 1, 1)
    block_dims = (10, 10, 1)
    result = matmul(A, B, C, dims)
    A_gpu = to_gpu(A)
    B_gpu = to_gpu(B)
    C_gpu = to_gpu(C)
    dims_gpu = to_gpu(dims)
    matmul_gpu(A_gpu, B_gpu, C_gpu, dims_gpu, block=block_dims, grid=grid_dims)
    cuda.memcpy_dtoh(C, C_gpu)
    assert False not in np.isclose(C, result), "Test case failed - matmul - {}".format(dims)
    print("Matmul Case Passed")

def test_matvec(A, B):
    dims = np.asarray(list(A.shape) + list(B.shape), dtype=np.int32)
    C = np.zeros((dims[0])).astype(np.float32)
    grid_dims = (1, 1, 1)
    block_dims = tuple([int(dims[0]), 1, 1])
    result = matvec(A, B, C, dims)
    A_gpu = to_gpu(A)
    B_gpu = to_gpu(B)
    C_gpu = to_gpu(C)
    dims_gpu = to_gpu(dims)
    matvec_gpu(A_gpu, B_gpu, C_gpu, dims_gpu, block=block_dims, grid=grid_dims)
    cuda.memcpy_dtoh(C, C_gpu)

    assert False not in np.isclose(C, result), "Test case failed - matvec - {}".format(dims)
    print("MatVec Case Passed")


def test_matadd_scalar(A, B):
    dims = np.asarray(list(A.shape) + [1, 1], dtype=np.int32)
    C = np.zeros(A.shape).astype(np.float32)
    grid_dims = (1, 1, 1)
    block_dims = tuple([int(dims[0]), int(dims[1]), 1])
    result = matadd_scalar(A, B, C, dims)
    A_gpu = to_gpu(A)
    B_gpu = np.dtype('float32').type(B)
    C_gpu = to_gpu(C)
    dims_gpu = to_gpu(dims)
    matadd_scalar_gpu(A_gpu, B_gpu, C_gpu, dims_gpu, block=block_dims, grid=grid_dims)
    cuda.memcpy_dtoh(C, C_gpu)

    assert False not in np.isclose(C, result), "Test case failed - matadd_scalar - {}".format(dims)
    print("MatAdd scalar Case Passed")

def test_matmul_scalar(A, B):
    dims = np.asarray(list(A.shape) + [1, 1], dtype=np.int32)
    C = np.zeros(A.shape).astype(np.float32)
    grid_dims = (1, 1, 1)
    block_dims = tuple([int(dims[0]), int(dims[1]), 1])
    result = matmul_scalar(A, B, C, dims)
    A_gpu = to_gpu(A)
    B_gpu = np.dtype('float32').type(B)
    C_gpu = to_gpu(C)
    dims_gpu = to_gpu(dims)
    matmul_scalar_gpu(A_gpu, B_gpu, C_gpu, dims_gpu, block=block_dims, grid=grid_dims)
    cuda.memcpy_dtoh(C, C_gpu)

    assert False not in np.isclose(C, result), "Test case failed - matmul_scalar - {}".format(dims)
    print("Matmul Scalar Case Passed")

    

def test_sumvec_inplace(A, B):
    assert A.shape[0] == B.shape[0], "mismatched vec sum dims"
    grid_dims = (1, 1, 1)
    block_dims = tuple([A.shape[0], 1, 1])
    result = sum_vec_inplace(A, B, A.shape[0])
    A_gpu = to_gpu(A)
    B_gpu = to_gpu(B)
    length = np.dtype('int32').type(A.shape[0])
    sum_vec_inplace_gpu(A_gpu, B_gpu, length, block=block_dims, grid=grid_dims)
    cuda.memcpy_dtoh(A, A_gpu)

    assert False not in np.isclose(A, result), "sumvec inplace - results not matched" 

    print("Sumvec Inplace case passed")

def test_sumvec_list(A):
    grid_dims = (1, 1, 1)
    block_dims = tuple([A.shape[1], A.shape[1], int(A.shape[0]/A.shape[1]) + 1])

    A_gpu = to_gpu(A)
    spacing = np.dtype('int32').type(1)
    length = np.dtype('int32').type(int(A.shape[0]))
    vec_size = np.dtype('int32').type(int(A.shape[1]))

    result = sumveclist(A, spacing, length, vec_size)
    sum_veclist_gpu(A_gpu, spacing, length, vec_size, block=block_dims, grid=grid_dims)

    cuda.memcpy_dtoh(A, A_gpu)

    assert False not in np.isclose(A[0], result), "sumvec list - results not matched"

    print("Sumveclist case passed")

    



mat_A_dims = (1, 1)
mat_B_dims = (2, 4)
mat_C_dims = (4, 6)
mat_D_dims = (3, 4)
mat_E_dims = (3, 4)
mat_F_dims = (2, 4)
mat_Z_dims = (20, 6)
mat_ZZ_dims = (1, 2)

vec_A_dims = (1)
vec_B_dims = (6)
vec_C_dims = (10)
vec_D_dims = (6)
vec_E_dims = (10)
vec_Z_dims = (2)

mat_A = np.random.rand(*mat_A_dims).astype(np.float32)
mat_B = np.random.rand(*mat_B_dims).astype(np.float32)
mat_C = np.random.rand(*mat_C_dims).astype(np.float32)
mat_D = np.random.rand(*mat_D_dims).astype(np.float32)
mat_E = np.random.rand(*mat_E_dims).astype(np.float32)
mat_F = np.random.rand(*mat_F_dims).astype(np.float32)
mat_Z = np.random.rand(*mat_Z_dims).astype(np.float32)
mat_ZZ = np.random.rand(*mat_ZZ_dims).astype(np.float32)

vec_A = np.random.rand(vec_A_dims).astype(np.float32)
vec_B = np.random.rand(vec_B_dims).astype(np.float32)
vec_C = np.random.rand(vec_C_dims).astype(np.float32)
vec_D = np.random.rand(vec_D_dims).astype(np.float32)
vec_E = np.random.rand(vec_E_dims).astype(np.float32)
vec_Z = np.random.rand(vec_Z_dims).astype(np.float32)


#dims = np.asarray(list(mat_A.shape) + list(mat_A.shape), dtype=np.int32)
#result = np.zeros((dims[0], dims[3])) 
#test_matmul(mat_A, vec_A, result, dims)

test_matmul(mat_B, mat_C)


test_matvec(mat_A, vec_A)
test_matvec(mat_C, vec_B)
test_matvec(mat_ZZ, vec_Z)


test_matadd_scalar(mat_A, 5)
test_matadd_scalar(mat_D, 3)
test_matadd_scalar(mat_B, 99)

test_matmul_scalar(mat_A, 3)
test_matmul_scalar(mat_D, 8)
test_matmul_scalar(mat_B, 100)

test_sumvec_inplace(vec_B, vec_D)
test_sumvec_inplace(vec_B, vec_B)
test_sumvec_inplace(vec_C, vec_E)

test_sumvec_list(mat_C)
test_sumvec_list(mat_Z)






