from enum import Enum
from functools import partial
import os

#processing type for the zonotope projection
class ZP_TYPE(Enum):
    CPU = 0        #standard cpu-only computation
    CPU_MP = 1      #multiprocessing enabled cpu-only computation
    GPU_Hybrid = 2  #mixed gpu/cpu implementation
    GPU = 3         #maximimally gpu oriented implementation



#todo should probably just have some class abstraction here
#proc_types are children overlaoding the setup() and verts()
# of 

def get_ZP_instance(zp_type):
    if not isinstance(zp_type, ZP_TYPE): 
        raise Exception("Invalid proc_type (undefined)")
    proc_type_class = {
        ZP_TYPE.CPU : CPU_ZP,
        ZP_TYPE.CPU_MP : CPU_MP_ZP,
        ZP_TYPE.GPU : GPU_ZP,
        ZP_TYPE.GPU_Hybrid : GPU_Hybrid_ZP
    }
    return proc_type_class[zp_type](zp_type)

class ZonoProcessor():
    def __init__(self, zp_type):
        self.type = zp_type
        self.setup()

    def setup(self):
        raise NotImplementedError("ZP setup fx must be defined")
    def verts(self, zonos):
        raise NotImplementedError("ZP verts fx must be defined")
    def verts_timeit(zonos):
        return self.verts(zonos)


class CPU_ZP(ZonoProcessor):
    def __init__(self, zp_type):
        super(CPU_ZP, self).__init__(zp_type)

    def setup(self):
        pass
    def verts(self, zonos):
        return [
            [a.tolist() for a in z.verts()] for z in list(zonos)
        ]

from quickzonoreach.zono import Zonotope, reload_environment
class CPU_MP_ZP(ZonoProcessor):
    def __init__(self, zp_type):
        from multiprocessing import Pool
        super(CPU_MP_ZP, self).__init__(zp_type)
        self.concurrency = 4
        self.process_pool = Pool(self.concurrency)


    def setup(self):
        pass
    def verts(self, zonos):
        return list(self.process_pool.map(Zonotope.verts, zonos))

from threading import Thread
class GPU_RUNNER(Thread):
    def __init__(self, threadID):
        Thread.__init__(self)
        self.threadID = threadID

    def run(self, zono):
        return zono.verts()


class GPU_Hybrid_ZP(ZonoProcessor):
    def __init__(self, zp_type):
        from multiprocessing.pool import ThreadPool #this is intended to speed up single-core execution so we use threads vs processes
        super(GPU_Hybrid_ZP, self).__init__(zp_type)
        self.concurrency = 4
        self.process_pool = ThreadPool(self.concurrency)
        os.environ["QZ_ENABLE_CUDA"] = "ENABLED" #force cuda enable for QZ
        reload_environment()
        #concurrent.futures is SIGNIFICANTLY slower (even slower than single core)


    def setup(self):
        pass
    def verts(self, zonos):
        return list(self.process_pool.map(Zonotope.verts, zonos)) #spawn process for each

class GPU_ZP(ZonoProcessor):
    def setup(self):
        pass
    def verts(self, zonos):
        pass


    
