from quickzonoreach import kamenev


def get_verts_gpu(dims, supp_point_func, epsilon=1e-7):
    '''
    get the n-dimensional vertices of the convex set defined through supp_point_func (which may be degenerate)
    '''

    init_simplex = _find_init_simplex(dims, supp_point_func)

    if len(init_simplex) < 3:
        return init_simplex # for 0-d and 1-d sets, the init_simplex corners are the only possible extreme points
    
    pts, _ = _v_h_rep_given_init_simplex_async_hybrid(init_simplex, supp_point_func, epsilon=epsilon)

    if dims == 2:
        # make sure verts are in order (for 2-d plotting)
        rv = []

        hull = ConvexHull(pts)

        #rv = hull.points
        max_y = -np.inf
        for v in hull.vertices:
            max_y = max(max_y, hull.points[v, 1])
            
            rv.append(hull.points[v])

        rv.append(rv[0])

    else:
        rv = pts

    return rv


#kamenev.get_verts = get_verts_gpu


def _v_h_rep_given_init_simplex_async_gpu(init_simplex, supp_point_func, epsilon=1e-7):
    # ----- CPU -> GPU MEMORY -----
    # VERTS - array with known max elements (how to find this?)
    # EPSILON - const float
    # HULL -- the computed convex hull
    # CURR_VERT -- index of newest set of verts
    # NUM_VERT -- total used verts in array
    # MAX_VERT -- max size of verts array
    # CENTER -- center of zonotope
    # INITIAL_BOUNDS -- initial zono bounds
    # ------ KERNELS ----
    # STREAM: supp_point_funx (maximize)
    # Fx: ConvexHull 

    #copy init_simplex to bottom of VERTS
    # ??? copy n copies of zonotope center to fill VERTS
    # copy initial_bounds to INITIAL_BOUNDS
    #copy epsilon value to EPSILON
    #initialize CURR_VERT to 0

    #start gpu kernel
    #https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf
    #overlap kernel and copy for new verts? ie. spawn copy operations in parallel to copy newly found verts between CURR_VERT and NUM_VERT
    #dynamic parallism to enable "nested" kernels (actually children more than nested I think)
    #use pycuda prepared_async_call for async kernel calling


    #CALL convexhull kernel, store result in HULL
    #CALL block to execute supp_point_fx with (NUM_VERT-CURR_VERT)=tcount threads
    #   using CURR_VERT as VERTS indexing offset...
    #   can bypass sep. kernel for max_func by directly indexing at VERTS[thread.idx+CURR_VERT](+1) instead of VERTS[0] and VERTS[1] for each
    #   spawn child kernels to transpose and store in locally scoped GPU mem
    #   spawn child kernels to perform dot products and store in GPU mem
    #   sum rowise product between transposed matrix and 
    #   compute error and compare with epsilon
    #       if so, copy new vertex to VERTS
    #       increment num_new_verts
    #   begin copy of VERTS[CURR_VERT, NUM_VERT] (device to host async)
    #   CURR_VERT, NUM_VERT += num_new_verts
    #   restart if num_new_verst was > 0


    #copy VERTS back to host memory
    pass


#hybrid gpu/cpu implementation
#convex hull is performed on CPU
def _v_h_rep_given_init_simplex_async_hybrid(init_simplex, supp_point_func, epsilon):
    new_pts = init_simplex
        
    verts = []
    iteration = 0
    max_error = None

    while new_pts:
        iteration += 1
        #print(f"\nIteration {iteration}. Verts: {len(verts)}, new_pts: {len(new_pts)}, max_error: {max_error}")
                
        first_new_index = len(verts)
        verts += new_pts

        hull = ConvexHull(verts)
        
        
        store = []
        #copy everything and call kernel
        for i, simplex in enumerate(hull.simplices):
            is_new = False

            for index in simplex:
                if index >= first_new_index:
                    is_new = True
                    break

            if not is_new:
                continue # skip this simplex

            # get hyperplane for simplex
            normal = hull.equations[i, :-1]
            rhs = -1 * hull.equations[i, -1]
            store.append((normal, rhs))

        normals, rhs = zip(*store)
        normals_np = np.array(normals).astype(np.float64)
        rhs_np = np.array(rhs).astype(np.float64)

        normals_GPU = gpuarray.to_gpu(normals_np)
        rhs_GPU = gpuarray.to_gpu(rhs_np)



        new_pts = []
        max_error = 0
        
        for normal, rhs in store:
            supporting_pt = supp_point_func(normal)
            
            error = np.dot(supporting_pt, normal) - rhs 
            max_error = max(max_error, error)

            assert error >= -1e-7, "supporting point was inside facet?"

            if error >= epsilon:
                # add the point... at this point points may be added twice... this doesn't seem to matter
                new_pts.append(supporting_pt)

    #points[hull.vertices]

    return np.array(verts, dtype=float), hull.equations
