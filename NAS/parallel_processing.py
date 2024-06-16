import multiprocessing
import multiprocessing.queues
from multiprocessing import Pool
from joblib import Parallel, delayed
import time

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

# def executeParallel(func, args, n_jobs, timeout):
#     results = []
#     for i in range(0, len(args), n_jobs):
#         argsToUse = args[i:min(i+n_jobs, len(args))]

#         queue = multiprocessing.Queue()
#         p = multiprocessing.Process(target=worker, args=(queue, func, argsToUse, n_jobs))
#         p.start()
#         p.join(timeout=timeout)

#         currentReults = queue.get_nowait()
#         results+=currentReults
#         if p.is_alive():
#             p.terminate()
#             p.join()
#     return results

def executeParallel(func, args, n_jobs, timeout):
    results = []

    def callback(result):
        results.append(result)
    
    for i in range(0, len(args), n_jobs):
        startTime = time.time()
        pool = Pool(processes=n_jobs)
        argsToUse = args[i:min(i+n_jobs, len(args))]
        [pool.apply_async(func, args=eachArgs, callback=callback) for eachArgs in argsToUse]
        time.sleep(timeout)
        print(time.time() - startTime)
        pool.terminate()
    return results