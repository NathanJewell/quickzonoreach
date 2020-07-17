

__device__ maximize(rv, mat_tp, init_bounds, rv_arr) {


}

    //def maximize(self, vector):
        //'get the maximum point of the zonotope in the passed-in direction'

        //rv = self.center.copy()

        //# project vector (a generator) onto row, to check if it's positive or negative
        //transpose = self.mat_t.transpose()
        //res_vec = np.dot(transpose, vector) # slow? since we're taking transpose

        //for res, row, ib in zip(res_vec, transpose, self.init_bounds):
            //factor = ib[1] if res >= 0 else ib[0]

            //rv += factor * row

        //return rv


__device__ void max_func(center, mat_t, vector, epsilon) {
    int xdim = 0;
    int ydim = 1;

    //max_func

    //maximize
    res_vec = vector * transpose //cublas matmul
    result = center;


}
            //def max_func(vec):
                //'projected max func for kamenev'

                //max_vec = [0] * dims
                //max_vec[xdim] += vec[0]
                //max_vec[ydim] += vec[1]
                //max_vec = np.array(max_vec, dtype=float)

                //res = self.maximize(max_vec)

                //return np.array([res[xdim], res[ydim]], dtype=float)

//max of #simplices new verts
__global__ void _v_h_rep_given_hull(
    float *center, int dims,
    float *mat_tp, int *mat_tp_dims, //zonotope member variables
    float *init_bounds, int *init_bounds_dims,
    float *verts,                         //all verts
    float *simplices, int *num_simplices, //convex hull simplices
    float *equations, int *num_equations, //convex hull equations
    float epsilon, //minimum error
    float error  //output variables
    ) {

    //block id determines which simplex is being looked at
    bool is_new = is_geq(simplices[threadId.x], first_new_vert);
    normal = simplices[threadId.x];
    rhs = simplices[threadId.x][num_simplices[threadId.x]-1];
    supp_pt = max_func(equations[threadId.x], num_simplices[threadId.x]-1, center, mat_t);

    error = dot(supp_pt, normal) - rhs;

    if (error >= epsilon) {
        new_verts[threadId.x] = supp_pt;
    }

    return;

}


//CODE TO BE PARALLELIZED

            //is_new = False

            //for index in simplex:
                //if index >= first_new_index:
                    //is_new = True
                    //break

            //if not is_new:
                //continue # skip this simplex

            //# get hyperplane for simplex
            //normal = hull.equations[i, :-1]
            //rhs = -1 * hull.equations[i, -1]
            //store.append((normal, rhs))

            //#COMPUTE NORMAL FOR ALL equations

            //supporting_pt = supp_point_func(normal)
            
            //error = np.dot(supporting_pt, normal) - rhs
            //max_error = max(max_error, error)

            //assert error >= -1e-7, "supporting point was inside facet?"

            //if error >= epsilon:
                //# add the point... at this point points may be added twice... this doesn't seem to matter
                //new_pts.append(supporting_pt)
