'''
Functions related to Kamenev's method for polytope approximation (the method of refined bounds)

Stanley Bak
May 16, 2019
'''

import numpy as np
from scipy.spatial import ConvexHull


def _get_orthonormal_rank(vecs, tol=1e-7):
    '''
    given a list of vecs, return a new vector orthonormal to them and the rank of the matrix
    '''

    _, s, v = np.linalg.svd(vecs)

    index = 0

    while index < len(s) and s[index] > tol:
        index += 1

    if index == len(v):
        rv_vec = None # the vectors span the space
    else:
        rv_vec = v[index]

    return rv_vec, index

def _get_rank(vecs, tol=1e-7):
    '''get the rank of the passed in matrix'''

    return _get_orthonormal_rank(vecs, tol=tol)[1]

def _find_two_points(dims, supp_point_func):
    '''find two points in the the convex set defined through supp_point_func (which may be degenerate)

    if len(pts) == 1, the convex set is a degenerate set consisting of a single pt
    '''

    pts = []

    for d in range(dims):
        vec = np.array([-1 if i == d else 0 for i in range(dims)], dtype=float)

        # try min
        p = supp_point_func(vec)
        assert len(p) == dims, f"support fuction returned {len(p)}-dimensional point, expected {dims}-d"

        if not pts:
            pts.append(p)
        elif not np.allclose(p, pts[0]):
            pts.append(p)
            break
        
        # try max
        vec = np.array([1 if i == d else 0 for i in range(dims)], dtype=float)
        p = supp_point_func(vec)

        if not np.allclose(p, pts[0]):
            pts.append(p)
            break

    return pts

def _find_init_simplex(dims, supp_point_func):
    '''
    find an n-dimensional initial simplex
    '''

    # first, construct the initial simplex and determine a basis for the convex set (it may be degenerate)
    init_simplex = _find_two_points(dims, supp_point_func)

    if len(init_simplex) == 2: # S may be a degenerate shape consisting of a single point
        init_vec = init_simplex[1] - init_simplex[0]

        spanning_dirs = [init_vec]
        degenerate_dirs = []
        vecs = [init_vec]

        for _ in range(dims - 1):
            new_dir, rank = _get_orthonormal_rank(vecs)

            # min/max in direction v, checking if it increases the rank of vecs
            pt = supp_point_func(new_dir)
            vecs.append(pt - init_simplex[0])

            if _get_rank(vecs) > rank:
                init_simplex.append(pt)
                spanning_dirs.append(vecs[-1])
                continue

            # rank did not increase with maximize, try minimize
            vecs = vecs[0:-1] # pop vec

            pt = supp_point_func(-1 * new_dir)
            vecs.append(pt - init_simplex[0])

            if _get_rank(vecs) > rank:
                init_simplex.append(pt)
                spanning_dirs.append(vecs[-1])
                continue

            # rank still didn't increase, new_dir is orthogonal to shape S
            vecs = vecs[0:-1] # pop vec

            vecs.append(new_dir) # forces a new orthonormal direction during the next iteration
            degenerate_dirs.append(new_dir)

    return init_simplex

def get_verts(dims, supp_point_func, epsilon=1e-7):
    '''
    get the n-dimensional vertices of the convex set defined through supp_point_func (which may be degenerate)
    '''

    init_simplex = _find_init_simplex(dims, supp_point_func)

    if len(init_simplex) < 3:
        return init_simplex # for 0-d and 1-d sets, the init_simplex corners are the only possible extreme points
    
    pts, _ = _v_h_rep_given_init_simplex(init_simplex, supp_point_func, epsilon=epsilon)

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

def get_verts_gpu(dims, supp_point_func, gpu_func, epsilon=1e-7):
    init_simplex = _find_init_simplex(dims, supp_point_func)

    pts, _ = _v_h_rep_given_init_simplex_gpu(init_simplex, gpu_func, epsilon=epsilon)

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


def _v_h_rep_given_init_simplex(init_simplex, supp_point_func, epsilon=1e-7):
    '''get all the vertices and hyperplanes of (an epsilon approximation of) the set, defined through supp_point_func

    This function is provided with an initial simplex which spans the space

    this returns verts, equations, where equations is from the Convex Hull's (hull.equations)
    '''

    new_pts = init_simplex
        
    verts = []
    iteration = 0
    max_error = None

    while new_pts:
        iteration += 1
        #print(f"\nIteration {iteration}. Verts: {len(verts)}, new_pts: {len(new_pts)}, max_error: {max_error}")
                
        first_new_index = len(verts)
        verts += new_pts
        new_pts = []
        max_error = 0

        hull = ConvexHull(verts)

        store = []
        for i, simplex in enumerate(hull.simplices):
            is_new = False

            for index in simplex:
                if index >= first_new_index:
                    is_new = True
                    break

            if not is_new:
                continue # skip this simplex

            #import pdb
            #pdb.set_trace()
            # get hyperplane for simplex
            normal = hull.equations[i, :-1]
            rhs = -1 * hull.equations[i, -1]
            store.append((normal, rhs))

            #COMPUTE NORMAL FOR ALL equations

            #import pdb
            #pdb.set_trace()
            supporting_pt = supp_point_func(normal)
            
            error = np.dot(supporting_pt, normal) - rhs
            max_error = max(max_error, error)

            assert error >= -1e-7, "supporting point was inside facet?"

            if error >= epsilon:
                # add the point... at this point points may be added twice... this doesn't seem to matter
                new_pts.append(supporting_pt)

    #points[hull.vertices]

    return np.array(verts, dtype=float), hull.equations

def _v_h_rep_given_init_simplex_gpu(init_simplex, gpu_func, epsilon=1e-7):
    from pycuda import gpuarray
    import pycuda.driver as cuda
        
    verts = np.asarray(init_simplex).astype(np.float32)
    #create array with space for n more amount of verts

    iteration = 0
    first_new_index = 0
    new_verts = None

    max_array_size = 100
    new_verts_empty = np.zeros((max_array_size, verts.shape[1]), dtype=np.float32)
    new_verts = new_verts_empty
    new_verts_GPU = cuda.mem_alloc(new_verts_empty.nbytes)
    simplices_GPU = cuda.mem_alloc(max_array_size * np.dtype(np.int32).itemsize)
    simplices_dims_GPU = cuda.mem_alloc(2 * np.dtype(np.int32).itemsize)
    equations_GPU = cuda.mem_alloc(max_array_size * np.dtype(np.float32).itemsize)
    equations_dims_GPU = cuda.mem_alloc(2 * np.dtype(np.int32).itemsize)

    while len(verts) > first_new_index:
        iteration += 1
        #print(f"\nIteration {iteration}. Verts: {len(verts)}, new_pts: {len(new_pts)}, max_error: {max_error}")
                
        first_new_index = len(verts)
        #copy verts from gpu

        hull = ConvexHull(verts)


        if len(hull.simplices) > max_array_size:
            raise RuntimeError #"CRITICAL ERROR, max_gpu_array_size is overrun"

        #copy empty data to vertices array
        cuda.memcpy_htod(new_verts_GPU, new_verts)

        #copy hull data to gpu asynchronously
        cuda.memcpy_htod(simplices_GPU, hull.simplices)
        cuda.memcpy_htod(simplices_dims_GPU, np.asarray(hull.simplices.shape).astype(np.int32))
        cuda.memcpy_htod(equations_GPU, hull.equations.astype(np.float32))
        cuda.memcpy_htod(equations_dims_GPU, np.asarray(hull.equations.shape).astype(np.int32))


        #spawn block and make device call for each simplex
        grid_dims = (len(hull.simplices), 1, 1)
        block_dims = ((10, 10, 1))

        #call cuda kernel for each stored (new) simplex IN: <simplices, first new index> OUT: <is_new_indices> 
        #find supporting points IN: <equations, center, transpose> OUT: <all verts supp pts>
        #product and add to find error <supp_point, jk
        #print("CALLED KERNEL")
        gpu_func(
            new_verts_GPU, 
            simplices_GPU, simplices_dims_GPU,
            equations_GPU, equations_dims_GPU,
            np.dtype('float32').type(epsilon),
            np.dtype('int32').type(first_new_index),
            block=block_dims, grid=grid_dims
        )

        #wait for hull data copy to finish
        #pycuda.wait_for_async
        cuda.memcpy_dtoh(new_verts, new_verts_GPU)
        new_verts = new_verts[~np.all(new_verts == 0, axis=1)] 
        verts = np.concatenate((verts, new_verts), axis=0) 


    #points[hull.vertices]
    new_verts_GPU.free()
    simplices_GPU.free()
    equations_GPU.free()

    return np.array(verts, dtype=float), hull.equations
