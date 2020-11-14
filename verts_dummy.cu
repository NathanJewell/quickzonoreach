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