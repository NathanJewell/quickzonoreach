
import pycuda.autoinit
from pycuda.compiler import DynamicSourceModule
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import skcuda.cublas as cublas

here = os.path.dirname(os.path.abspath(__file__))
kernel_filename = "verts_test.cu" 
kernel_filepath = os.path.join(here, kernel_filename)
with open(kernel_filepath, "r") as kernel_file:
    kernel_source_string = '\n'.join(kernel_file.readlines())
    module = DynamicSourceModule(kernel_source_string, options=['-g'])
    
    matmul_gpu = module.get_function("matmul_global")
    matvec_gpu = module.get_function("matvec_global")
    matadd_scalar_gpu = module.get_function("matadd_scalar_global")
    matmul_scalar_gpu = module.get_function("matmul_scalar_global")
    sumveclist_gpu = module.get_function("sum_vec_list_global")

import numpy as np
def matmul(A, B, C, dims):
    return np.matmul(A, B)

def matvec(A, B, C, dims):
    return np.dot(A, B)

def matadd_scalar(A, B, C, dims):
    return A + B

def matmul_scalar(A, B, C, dims):
    return A * B

def sumveclist(A, spacing, length, vec_size):
    pass


def to_gpu(obj, type):
    obj_gpu = cuda.mem_alloc(obj.nbytes)
    cuda.memcpy_htod(obj_gpu, obj)
    return obj_gpu

def test_matmul(A, B, C, dims):


def test_matvec():


def test_matadd_scalar():

def test_matmul_scalar():

def test_sumveclist():


mat_A_dims = (1, 1)
mat_B_dims = (2, 2)
mat_C_dims = (4, 6)
mat_D_dims = ()

vec_A_dims = ()
vec_B_dims = ()

mat_A = np.random.rand(mat_A_dims)
mat_B = np.random.rand(mat_B_dims)
mat_C = np.random.rand(mat_C_dims)
mat_D = np.random.rand(mat_D_dims)

    new_verts_GPU = cuda.mem_alloc(new_verts_empty.nbytes)
    simplices_GPU = cuda.mem_alloc(max_array_size * np.dtype(np.int32).itemsize)
    simplices_dims_GPU = cuda.mem_alloc(2 * np.dtype(np.int32).itemsize)
    equations_GPU = cuda.mem_alloc(max_array_size * np.dtype(np.float32).itemsize)
    equations_dims_GPU = cuda.mem_alloc(2 * np.dtype(np.int32).itemsize)


    cuda.memcpy_htod(simplices_GPU, hull.simplices)
    cuda.memcpy_htod(simplices_dims_GPU, hull.simplices.shape)
    cuda.memcpy_htod(equations_GPU, hull.equations.astype(np.float32))
    cuda.memcpy_htod(equations_dims_GPU, hull.equations.shape)