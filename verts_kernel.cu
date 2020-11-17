// AUTHOR: Nathan Jewell
// July 2020


//device fx to perform matrix product on single 2d block
//dims = { A-rows, A-cols, B-rows, B-cols}
//so the output dimension should be A-rows x B-cols
__device__ void matmul ( 
    float *A, float *B, float *C,
    int* dims
) {
    int row = threadIdx.x;
    int col = threadIdx.y;

    if ( col >= dims[3] || row >= dims[0] || threadIdx.z != 0) {
        //thread is out of bounds - will not execute anything
        return;
    }

    float idx_sum = 0;
    for (int k = 0; k < dims[1]; k++) {
        idx_sum += A[row*dims[1] + k] * B[k * dims[3] + col];
    }

    C[row * dims[3] + col] = idx_sum;
}

__global__ void matmul_global ( 
    float *A, float *B, float *C,
    int* dims
) {
    matmul(A, B, C, dims);
}

//multiply nd matrix by vector in style of np.dot
__device__ void matvec (
    float *A, float *B, float *C, int *dims
) {
    int row = threadIdx.x;

    if ( row >= dims[0] || (threadIdx.y | threadIdx.z) != 0 ) {
        //thread is out of bounds - will not execute anything
        return;
    }

    float idx_sum = 0;
    for (int k = 0; k < dims[1]; k++) {
        idx_sum += A[row*dims[1] + k] * B[k];
    }
    C[row] = idx_sum;
}

__global__ void matvec_global (
    float *A, float *B, float *C, int *dims
) {
    matvec(A, B, C, dims);
}
//device fx to perform matrix scalar multiplaction on single 2d thread block
//dims = {A rows, A cols}
__device__ void matadd_scalar (
    float *A, float B, float * C, int * dims
) {

    int row = threadIdx.x;
    int col = threadIdx.y;
    if ( row >= dims[0] || col >= dims[1] || threadIdx.z != 0) {
        //thread is out of bounds - will not execute anything
        return;
    }

    C[row * dims[1] + col] = A[row * dims[1] + col] + B;
}

__global__ void matadd_scalar_global (
    float *A, float B, float * C, int * dims
) {
    matadd_scalar(A, B, C, dims);
}

__device__ void matmul_scalar (
    float *A, float B, float *C, int * dims
) {
    int row = threadIdx.x;
    int col = threadIdx.y;
    if ( row >= dims[0] || col >= dims[1] || threadIdx.z != 0) {
        //thread is out of bounds - will not execute anything
        return;
    }

    C[row * dims[1] + col] = A[row * dims[1] + col] * B;
}

__global__ void matmul_scalar_global (
    float *A, float B, float *C, int * dims
) {
    matmul_scalar(A, B, C, dims);
}

__device__ void vecmul_scalar (
    float *A, float B, float *C, int * dims
) {
    int col = threadIdx.x;
    if (col >= dims[1] ) {
        //thread is out of bounds - will not execute anything
        return;
    }

    C[col] = A[col] * B;
}

__global__ void vecmul_scalar_global (
    float *A, float B, float *C, int * dims
) {
    vecmul_scalar(A, B, C, dims);
}

//A and B must have same dims
__device__ void sum_vec_inplace(float *A, float *B, int vec_size) {
    int col = threadIdx.x;
    if (col < vec_size) {
        A[col] = A[col] + B[col];
    }
}

__global__ void sum_vec_inplace_global(float * A, float *B, int vec_size) {
    sum_vec_inplace(A, B, vec_size);
}

//simple distributed vector list sum
__device__ void sum_vec_list (
    float *A, int spacing, int length, int vec_size
) {
//vecsize is length of vectors within the list
//spacing is the number of vectors between result vectors after this operation
//length is the total number of entries
    int list_idx = threadIdx.z * blockDim.y + threadIdx.y;
    if (list_idx >= int(length / spacing)) return; //threadIdx.x and z represents location within list
    if (threadIdx.x >= vec_size) return; //threadIdx.y represents location within vector
    int node = list_idx * spacing * vec_size; //location of this sums result
    int other = node + (spacing) * vec_size;
    if (other >= (length * vec_size)) return; //A[node] = A[node]
    sum_vec_inplace(&A[node], &A[other], vec_size);
    //A[node] = A[node] + A[node + spacing * vec_size];
    __threadfence();
    if (spacing <= (length/2) + 1) {
        sum_vec_list(A, spacing*2, length, vec_size);
    }
}

__global__ void sum_vec_list_global (
    float *A, int spacing, int length, int vec_size
) {
    sum_vec_list(A, spacing, length, vec_size);
}

__global__ void find_supp_point(
    float *center, int dims,
    float *mat_tp, int *mat_tp_dims, //zonotope member variables
    float *init_bounds, int *init_bounds_dims,
    float *new_verts,             //output space for supp points which are new vertices
    float *simplices, int *simplices_dims, //convex hull simplices
    float *equations, int *equations_dims, //convex hull equations
    float epsilon, //minimum error
    int first_new_index
    ) {

    //block id determines which simplex is being looked at
    __shared__ float normal[2];
    __shared__ float rhs;
    __shared__ float * res_vec;
    __shared__ float * max_vec; 
    __shared__ float * rv_list;
    __shared__ bool is_new;

    int row = blockIdx.x;
    int idx = threadIdx.y * gridDim.x + threadIdx.x;
    int xdim = 0; int ydim = 1;

    //check if the simplex (per block) is new
    if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
        if (blockIdx.x >= simplices_dims[0]) return;
        is_new = false; 

        for (int k = 0; k < simplices_dims[1]; k++){
            if (simplices[row*simplices_dims[1] + k] >= first_new_index) {
                is_new = true;
                break;
            }
        }

        if (!is_new) return;

        //setup some block-shared variables
        int eq_idx = row * equations_dims[0];
        normal[0] = equations[eq_idx];
        normal[1] = equations[eq_idx+1]; //all but last element in row
        rhs = -1 * equations[eq_idx + 2]; //-1 * last element in row

        res_vec = (float *)malloc( sizeof(float) * mat_tp_dims[0] * 1 );
        max_vec = (float *)malloc(dims * sizeof(float));
        rv_list = (float *)malloc(((mat_tp_dims[0]+1) * mat_tp_dims[1])* sizeof(float));
        memset(max_vec, 0, dims*sizeof(float));
        memcpy(rv_list, center, dims * sizeof(float));
        max_vec[xdim] = normal[0];
        max_vec[ydim] = normal[1];
        
    }

    //sync threads so we can increase concurrency
    __syncthreads();
    if (!is_new) return; //nothing further in this block - allready computed

    int combined_dims[4] = { mat_tp_dims[0], mat_tp_dims[1], dims, 1};
    //matvec(mat_tp, max_vec, res_vec, combined_dims);
    matvec(mat_tp, max_vec, res_vec, combined_dims); //first elem of rv_list is rv
    //res_vec is matrix of shape (mat_tp_dims[0] x 1)

    __syncthreads();

    int rv_row = threadIdx.z * blockDim.y + threadIdx.y;
    if (rv_row < mat_tp_dims[0]) {
        float factor; 
        if (res_vec[rv_row] >= 0) {
            factor = init_bounds[rv_row * init_bounds_dims[1] + 1];
        } else {
            factor = init_bounds[rv_row * init_bounds_dims[1]];
        }
        //1d vector by scalar
        combined_dims[0] = 1; combined_dims[1] = mat_tp_dims[1];
        combined_dims[2] = 1;combined_dims[3] = mat_tp_dims[1];
        vecmul_scalar(&mat_tp[rv_row * mat_tp_dims[1]], factor, &rv_list[(rv_row+1)*mat_tp_dims[1]], combined_dims);
    }
    //append center to rv_list
    //sum rv_list into res_vec
    __syncthreads();

    int spacing = 1;
    sum_vec_list(rv_list, spacing, mat_tp_dims[0] + 1, mat_tp_dims[1]);

    //while ( spacing <= ((mat_tp_dims[0]/2) + 1)) {
        //sum_vec_list(rv_list, spacing, mat_tp_dims[0] + 1, mat_tp_dims[1]);
        //spacing = spacing * 2;
        //__threadfence();
    //}

    __threadfence();

    combined_dims[0] = 1; combined_dims[1] = mat_tp_dims[0];
    combined_dims[2] = mat_tp_dims[0] ;combined_dims[3] = 1;

    matvec(&rv_list[0], &normal[0], res_vec, combined_dims);

    __threadfence();

    combined_dims[0] = 1; combined_dims[1] = mat_tp_dims[0];
    combined_dims[2] = normal[0];combined_dims[3] = 1;
    matadd_scalar(res_vec, -1 * rhs, res_vec, combined_dims); //operated in-place on input vector

    __threadfence();
    //add the point if it is in the face
    if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
        if (res_vec[0] > 0.000000) {
            new_verts[row*2] = rv_list[xdim];
            new_verts[row*2+1] = rv_list[ydim];
        }
    }
    //free(max_vec);
    //free(res_vec);
    //free(rv_list);
    return;

}

__global__ void dummy_supp_point(
    float *center, int dims,
    float *mat_tp, int *mat_tp_dims, //zonotope member variables
    float *init_bounds, int *init_bounds_dims,
    float *new_verts,             //output space for supp points which are new vertices
    float *simplices, int *simplices_dims, //convex hull simplices
    float *equations, int *equations_dims, //convex hull equations
    float epsilon, //minimum error
    int first_new_index
    ) {

    //block id determines which simplex is being looked at
    int row = blockIdx.x * 2;
    int xdim = 0; int ydim = 1;

    //once per block
    if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
        //add the point if it is in the face
        new_verts[row] = blockIdx.x;
        new_verts[row+1] = blockIdx.y;
    }

    return;

}