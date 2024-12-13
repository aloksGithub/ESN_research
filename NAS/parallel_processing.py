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
        pool = Pool(processes=n_jobs)
        argsToUse = args[i:min(i+n_jobs, len(args))]
        [pool.apply_async(func, args=eachArgs, callback=callback) for eachArgs in argsToUse]
        time.sleep(timeout)
        pool.terminate()
    return results

def executeParallelBatch(func, args, batchSize, timeout):
    results = []
    for i in range(0, len(args), batchSize):
        results+=executeParallelImproved(func, args[i:i+batchSize], batchSize, timeout)
    return results

def executeParallelImproved(func, args, n_jobs, timeout):
    """
    Executes functions in parallel with improved error handling:
    - Returns None for any failed jobs
    - Maintains argument order in results
    - Exits early if all processes finish before timeout
    """
    with Pool(processes=n_jobs) as pool:
        async_results = []
        for arg in args:
            async_result = pool.apply_async(func, args=arg)
            async_results.append(async_result)
        
        # Wait for all tasks to complete or timeout
        start_time = time.time()
        results = [None] * len(args)
        completed = [False] * len(args)
        
        while not all(completed) and time.time() - start_time < timeout:
            for i, ar in enumerate(async_results):
                if not completed[i] and ar.ready():
                    try:
                        results[i] = ar.get(timeout=0)
                        completed[i] = True
                    except:
                        results[i] = None
                        completed[i] = True
            time.sleep(0.1)  # Short sleep to prevent busy waiting
        
        # If there are any remaining tasks, terminate them
        if not all(completed):
            pool.terminate()
        else:
            pool.close()
        
        pool.join()
    
    return results