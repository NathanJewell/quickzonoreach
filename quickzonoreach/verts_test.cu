// AUTHOR: Nathan Jewell
// July 2020


//device fx to perform matrix product on single 2d thread block
__device__ float * matmul ( 
        float *A, float *B, 
        int* dims
    ) {
        int row = blockIdx.x*blockDim.x + threadIdx.x;
        int col = blockIdx.y*blockDim.y + threadIdx.y;

        float * C = (float *)malloc(dims[0] * dims[1] * sizeof(float));

        float idx_sum = 0;
        if (row < dims[2] && col < dims[1]) {
            for (int k = 0; k < dims[0]; k++) {
                idx_sum += A[row*dims[0] + k] * B[k * dims[1] + col];
            }
        }

        C[row * dims[0] + col] = idx_sum;
        return C;
}

//device fx to perform matrix scalar multiplaction on single 2d thread block
__device__ float * matadd_scalar (
    float *A, float B, int * dims
) {
    return A;
}

//device fx to compute supporting pt from zono and hull normal
__device__ float * find_supp_pt (
    float * normal, float * center, int dims, float * mat_tp, int * mat_tp_dims
) {
    int xdim = 0; int ydim = 1;
    float *max_vec = (float *)malloc(dims * sizeof(float));
    memset(max_vec, 0, dims*sizeof(float));

    max_vec[xdim] = normal[0];
    max_vec[ydim] = normal[1];

    float * supp_pt = (float *)malloc( sizeof(center) );
    //memcpy(center, new_pt, sizeof(center));

    int output_dims[4] = {2,3,4, 5};
    float * new_pt = matmul(normal, mat_tp, output_dims);

    //dealloc(new_pt);
    free(max_vec);
    return normal;
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

int row = blockIdx.x * 2;

    //check if the simplex (per block) is new
    if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
        bool is_new = false; 

        for (int k = 0; k < init_bounds_dims[1]; k++){
            if (simplices[row + k] >= first_new_index) {
                is_new = true;
                break;
            }
        }

        if (!is_new) return;

        float supp_pt[2] = {blockIdx.x, blockIdx.y};
        new_verts[row] = supp_pt[0];
        new_verts[row+1] = supp_pt[1];
    }

    //sync threads so we can increase concurrency
    //__syncthreads();

    //float normal[2] = { equations[row], equations[row+1]};
    //float rhs = -1 * equations[row + 1];
    //float * supp_pt = find_supp_pt(&normal[0], center, dims, mat_tp, mat_tp_dims);

    //__syncthreads();

    //int output_dims[4] = {2,3,4, 5};
    //float * error = matadd_scalar(matmul(supp_pt, &normal[0], output_dims),  -1 * rhs, output_dims);

    //add the point if it is in the facet
    //if (*error >= epsilon) {
        //new_verts[row] = supp_pt[0];
        //new_verts[row+1] = supp_pt[1];
    //}


    return;

}