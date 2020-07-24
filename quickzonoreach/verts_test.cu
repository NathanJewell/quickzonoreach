
//max of #simplices new verts
__global__ void find_supp_point(
    float *center, int dims,
    float **mat_tp, int *mat_tp_dims, //zonotope member variables
    float **init_bounds, int *init_bounds_dims,
    float *verts, int num_verts,             //all verts
    float **simplices, //int *simplices_dims, //convex hull simplices
    float **equations, //int *equations_dims, //convex hull equations
    float epsilon //minimum error
    ) {

    //block id determines which simplex is being looked at
    //bool is_new = is_geq(simplices[threadId.x], first_new_vert);

    //if (!is_new) {
        //return;
    //}

    //normal = equations[threadId.x];
    //rhs = equations[threadId.x][num_equations[threadId.x]-1];
    //supp_pt = max_func(normal, center, mat_t);

    //error = dot(supp_pt, normal) - rhs;

    //if (error >= epsilon) {
        //new_verts[threadId.x] = supp_pt;
    //}

    //verts[num_verts + blockId.x] = [blockId.x, 0]

    return;

}