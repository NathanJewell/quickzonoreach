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

    if ( col >= dims[1] || row >= dims[2] ) {
        //thread is out of bounds - will not execute anything
        return;
    }

    float idx_sum = 0;
    for (int k = 0; k < dims[1]; k++) {
        idx_sum += A[row*dims[1] + k] * B[k * dims[3] + col];
    }

    C[row * dims[1] + col] = idx_sum;
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

    if ( row >= dims[0] || threadIdx.y != 0 ) {
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
    if ( col >= dims[0] || row >= dims[1] ) {
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
    if ( col >= dims[0] || row >= dims[1] ) {
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
//simple distributed vector list sum
__device__ void sum_vec_list (
    float *A, int spacing, int length, int vec_size
) {
    if (threadIdx.x >= int(length / spacing)) return; //threadIdx.x represents location within list
    if (threadIdx.y >= vec_size) return; //threadIdx.y represents location within vector
    int node = threadIdx.x * spacing * vec_size + threadIdx.y;
    int other = node + spacing * vec_size;
    if (other >= length) return; //A[node] = A[node]
    A[node] = A[node] + A[node + spacing * vec_size];
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
        is_new = false; 

        for (int k = 0; k < init_bounds_dims[1]; k++){
            if (simplices[row + k] >= first_new_index) {
                is_new = true;
                break;
            }
        }

        if (!is_new) return;

        //setup some block-shared variables
        int eq_idx = row * simplices_dims[1];
        float normal[2] = { equations[eq_idx], equations[eq_idx+1]}; //all but last element in row
        float rhs = -1 * equations[eq_idx + 2]; //-1 * last element in row

        res_vec = (float *)malloc( sizeof(float) * mat_tp_dims[0] * 1 );
        max_vec = (float *)malloc(dims * sizeof(float));
        rv_list = (float *)malloc((mat_tp_dims[0]+1) * mat_tp_dims[1])* sizeof(float));
        memset(max_vec, 0, dims*sizeof(float));
        memcpy(res_vec, center, dims * sizeof(float));
        max_vec[xdim] = normal[0];
        max_vec[ydim] = normal[1];
        
    }

    //sync threads so we can increase concurrency
    __syncthreads();
    if (!is_new) return; //nothing further in this block - allready computed

    int combined_dims[4] = { mat_tp_dims[0], mat_tp_dims[1], dims, 1};
    //matvec(mat_tp, max_vec, res_vec, combined_dims);
    matvec(mat_tp, max_vec, rv_list, combined_dims); //first elem of rv_list is rv
    //res_vec is matrix of shape (mat_tp_dims[0] x 1)

    __syncthreads();

    if (threadIdx.x < mat_tp_dims[0]) {
        float factor; 
        if (res_vec[threadIdx.x] >= 0) {
            factor = init_bounds[threadIdx.x * init_bounds_dims[1] + 1];
        } else {
            factor = init_bounds[threadIdx.x * init_bounds_dims[1]];
        }
        combined_dims = {mat_tp_dims[1], 1, 1, 1}; //1d vector by scalar
        matmul_scalar(mat_tp[threadIdx.x * mat_tp_dims[1]], factor, rv_list[threadId.x*mat_tp_dims[1]+1], combined_dims);
    }
    //append center to rv_list
    //sum rv_list into res_vec
    __syncthreads();
    int spacing = 2;
    while ( spacing <= (mat_tp_dims[0]/2)) {
        sum_vec_list(rv_list, mat_tp_dims[1] + 1, mat_tp_dims[1]);
        __syncthreads();
        spacing = spacing * 2;
    }

    
    __syncthreads();

    combined_dims = {res_vec_, };

    matmul(rv_list[0], &normal[0], res_vec, combined_dims);
    matadd_scalar(res_vec, -1 * rhs, res_vec, combined_dims); //operated in-place on input vector

    //add the point if it is in the face
    if (*res_vec[0] >= epsilon) {
        new_verts[row] = rv_list[xdim];
        new_verts[row+1] = rv_list[ydim];
    }

    free(max_vec);
    free(res_vec);
    free(rv_list);
    return;

}