from enum import Enum
from functools import partial
import os
import time

#processing type for the zonotope projection
class ZP_TYPE(Enum):
    CPU = 0        #standard cpu-only computation
    CPU_MP = 1      #multiprocessing enabled cpu-only computation
    GPU_DUMMY = 2
    GPU_HYBRID= 3  #mixed gpu/cpu implementation
    GPU = 4         #maximimally gpu oriented implementation



#todo should probably just have some class abstraction here
#proc_types are children overlaoding the setup() and verts()
# of 

instance_dict = {}

def get_ZP_instance(zp_type):
    if not isinstance(zp_type, ZP_TYPE): 
        raise Exception("Invalid proc_type (undefined)")
    proc_type_class = {
        ZP_TYPE.CPU : CPU_ZP,
        ZP_TYPE.CPU_MP : CPU_MP_ZP,
        ZP_TYPE.GPU_DUMMY : GPU_DUMMY_ZP,
        ZP_TYPE.GPU_HYBRID: GPU_HYBRID_ZP,
        ZP_TYPE.GPU : GPU_ZP,
    }
    if not zp_type in instance_dict:
        instance_dict[zp_type] = proc_type_class[zp_type](zp_type)
    return instance_dict[zp_type]

class ZonoProcessor():
    def __init__(self, zp_type):
        self.type = zp_type
        self.setup()
        self.timing = None

    def setup(self):
        raise NotImplementedError("ZP setup fx must be defined")
    def verts(self, zonos):
        raise NotImplementedError("ZP verts fx must be defined")
    def verts_timeit(self, zonos):
        start = time.time()
        result = self.verts(zonos)
        elapsed = time.time() - start
        self.timing = elapsed
        return elapsed


class CPU_ZP(ZonoProcessor):
    def __init__(self, zp_type):
        super(CPU_ZP, self).__init__(zp_type)

    def setup(self):
        pass
    def verts(self, zonos):
        return [
            [a.tolist() for a in z.verts_cpu()] for z in list(zonos)
        ]

from quickzonoreach.zono import Zonotope, reload_environment
class CPU_MP_ZP(ZonoProcessor):
    def __init__(self, zp_type):
        from multiprocessing import Pool
        super(CPU_MP_ZP, self).__init__(zp_type)
        os.environ["QZ_ENABLE_CUDA"] = "DISABLED" #force cuda enable for QZ
        reload_environment()
        self.concurrency = 4
        self.process_pool = Pool(self.concurrency)


    def setup(self):
        pass
    def verts(self, zonos):
        return list(self.process_pool.map(Zonotope.verts_cpu, zonos))

#class GPUThread(threading.Thread):
    #def __init__(self, number):

class GPU_DUMMY_ZP(ZonoProcessor):
    def __init__(self, zp_type):
        super(GPU_DUMMY_ZP, self).__init__(zp_type)
        os.environ["QZ_ENABLE_CUDA"] = "ENABLED" #force cuda enable for QZ
        reload_environment(dummy=True)

    def setup(self):
        pass
    def verts(self, zonos):
        return [z.verts_gpu() for z in zonos]

class GPU_HYBRID_ZP(ZonoProcessor):
    def __init__(self, zp_type):
        super(GPU_HYBRID_ZP, self).__init__(zp_type)
        #try:
            #multiprocessing.set_start_method("spawn")
            #print("Threads now spawn not fork.")
        #except RuntimeError:
            #pass
        os.environ["QZ_ENABLE_CUDA"] = "ENABLED" #force cuda enable for QZ
        reload_environment()
        #concurrent.futures is SIGNIFICANTLY slower (even slower than single core)


    def setup(self):
        pass
    def verts(self, zonos):
        return [z.verts_gpu() for z in zonos]

class GPU_ZP(ZonoProcessor):
    pass
    #raise NotImplementedError("ZP not implemented")


    
