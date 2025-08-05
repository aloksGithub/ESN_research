import multiprocessing
import multiprocessing.queues
from multiprocessing import Pool
import time
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


def executeParallel(func, args, n_jobs, timeout):
    """
    DEPRECATED: This function is unsafe and inefficient. It uses a blocking
    `time.sleep()` and can terminate processes mid-calculation, leading to
    lost work and potential errors. Please use `executeParallelBatch` instead.
    """
    results = []

    def callback(result):
        results.append(result)

    for i in range(0, len(args), n_jobs):
        pool = Pool(processes=n_jobs)
        argsToUse = args[i : min(i + n_jobs, len(args))]
        [
            pool.apply_async(func, args=eachArgs, callback=callback)
            for eachArgs in argsToUse
        ]
        time.sleep(timeout)
        pool.terminate()
    return results


def executeParallelBatch(func, args, batchSize, timeout):
    results = []
    for i in range(0, len(args), batchSize):
        results += executeParallelImproved(
            func, args[i : i + batchSize], batchSize, timeout
        )
    return results


def executeParallelImproved(func, args, n_jobs, timeout):
    """
    Executes functions in parallel with improved error handling:
    - Returns None for any failed jobs
    - Maintains argument order in results
    - Exits early if all processes finish before timeout
    """
    with Pool(processes=n_jobs) as pool:
        async_results = [pool.apply_async(func, args=arg) for arg in args]

        results = [None] * len(args)

        start_time = time.time()
        for i, ar in enumerate(async_results):
            try:
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time < 0:
                    remaining_time = 0
                results[i] = ar.get(timeout=remaining_time)
            except Exception as e:
                # IMPROVEMENT: Instead of terminating the pool, we now just log
                # the failure and continue, allowing other jobs to complete.
                print(f"Warning: Job {i} failed with error: {e}")
                results[i] = None

    return results
