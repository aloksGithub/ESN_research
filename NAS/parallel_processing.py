import multiprocessing
from joblib import Parallel, delayed

def worker(queue, func, args, n_jobs):
    dataStore = [None] * len(args)
    queue.put(dataStore)
    def funcWrapper(jobIndex, *args):
        result = func(*args)
        dataStore[jobIndex] = result
        while not queue.empty():
            queue.get() # Call get on queue to empty it
        queue.put(dataStore)
        
    parallel = Parallel(n_jobs=n_jobs, require='sharedmem')
    parallel(delayed(funcWrapper)(i, *args[i]) for i in range(len(args)))

def executeParallel(func, args, n_jobs, timeout):
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=worker, args=(queue, func, args, n_jobs))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        result = queue.get_nowait()
        p.terminate()
        p.join()
        return result
    result = queue.get_nowait()
    return result