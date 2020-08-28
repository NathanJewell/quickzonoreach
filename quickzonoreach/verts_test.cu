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

    C[row * dims[1] + col] = A[row * dims[1] + col] + B
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

    C[row * dims[1] + col] = A[row * dims[1] + col] * B
}

__global__ void find_supp_point(
    float *center, int dims,
    float *mat_tp, int *mat_tp_dims, //zonotope member variables
    float *init_bounds, int *init_bounds_dims,
    float *new_verts,             //output space for supp points which are new vertices
    float *simplices, //int *simplices_dims, //convex hull simplices
    float *equations, //int *equations_dims, //convex hull equations
    float epsilon, //minimum error
    int first_new_index
    ) {

    //block id determines which simplex is being looked at
    __shared__ float normal[2];
    __shared__ float rhs;
    __shared__ float * res_vec;
    __shared__ float * max_vec; 
    __shared__ bool is_new;
    int row = blockIdx.x * 2;
    int idx = threadIdx.y * gridDim.x + threadIdx.x
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
        normal[2] = { equations[row], equations[row+1]};
        rhs = -1 * equations[row + 1];

        memset(max_vec, 0, dims*sizeof(float));
        memcpy(rv, center, sizeof(float) * dims)
        res_vec = (float *)malloc( sizeof(float) * mat_tp_dims[1] * 1 );
        max_vec = (float *)malloc(dims * sizeof(float));
        max_vec[xdim] = normal[0];
        max_vec[ydim] = normal[1];
        
    }

    //sync threads so we can increase concurrency
    __syncthreads();
    if (!is_new) return; //nothing further in this block - allready computed

    __syncthreads();
    //colpy center to rv
    combined_dims = { mat_tp_dims[0], mat_tp_dims[1], 1, dims}
    matmul(mat_tp, max_vec, res_vec, output_dims);

    //dims of res vec are: (mat_tp_dims[1], 1)

    if (threadIdx.x < mat_tp_dims[1]) {
        float factor; 
        if (res_vec[threadIdx.x] >= 0) {
            factor = init_bounds[threadIdx.x * init_bounds_dims[1] + 1]
        } else {
            factor = init_bounds[threadIdx.x * init_bounds_dims[1]]
        }
        combined_dims = {}
        rv_list[threadIdx.x] = matmul_scalar(mat_tp[threadIdx.x * mat_tp_dims[1]], )
    }
    //sum rv_list
    //copy center to res_vec
    //add sum to center


    
    __syncthreads();

    combined_dims = {2,3,4, 5};

    matmul(res_vec, &normal[0], output_dims)
    matadd_scalar(res_vec, -1 * rhs, output_dims); //operated in-place on input vector

    //add the point if it is in the face
    if (*error >= epsilon) {
        new_verts[row] = rv[xdim]
        new_verts[row+1] = rv[ydim]
    }

    free(max_vec);
    free(res_vec);
    return;

}