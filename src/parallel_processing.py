import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, wait
import math
import dill
import tempfile
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


def executeParallelBatch(func, args, batchSize, timeout, log_level=2):
    results = []
    for i in range(0, len(args), batchSize):
        results += executeParallelImproved(
            func, args[i : i + batchSize], batchSize, timeout, log_level
        )
    return results


def executeParallelImproved(func, args, n_jobs, timeout, log_level=2):
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
            except multiprocessing.TimeoutError:
                # Explicitly report timeouts; string for TimeoutError is often empty
                total_wait = time.time() - start_time
                if log_level >= 2:
                    print(
                        f"Warning: Job {i} timed out after waiting {total_wait:.2f}s (budget {timeout}s)"
                    )
                results[i] = None
            except Exception as e:
                # IMPROVEMENT: Instead of terminating the pool, we now just log
                # the failure and continue, allowing other jobs to complete.
                exc_name = type(e).__name__
                if log_level >= 1:
                    print(f"Warning: Job {i} failed with error: {exc_name}: {e}")
                results[i] = None

    return results


def _batch_worker(func, batch_args, max_workers, batch_timeout, result_file):
    """Runs inside a child process. Uses threads for JAX GPU parallelism.

    All threads share a single CUDA context within this process.
    Results are saved to a pickle file so the parent can read them
    after terminating this process to free GPU memory.
    """
    # Must be set before JAX/XLA is imported by func
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    batch_results = [None] * len(batch_args)
    errors = []  # (index, error_string) pairs

    executor = ThreadPoolExecutor(max_workers=max_workers)
    future_to_idx = {}
    for i, arg in enumerate(batch_args):
        future = executor.submit(func, *arg)
        future_to_idx[future] = i

    done, not_done = wait(future_to_idx.keys(), timeout=batch_timeout)

    for future in done:
        idx = future_to_idx[future]
        try:
            batch_results[idx] = future.result(timeout=0)
        except Exception as e:
            errors.append((idx, f"{type(e).__name__}: {e}"))

    timed_out_indices = [future_to_idx[f] for f in not_done]

    executor.shutdown(wait=False, cancel_futures=True)

    try:
        with open(result_file, 'wb') as f:
            dill.dump({
                'results': batch_results,
                'errors': errors,
                'timed_out': timed_out_indices,
            }, f)
    finally:
        # ThreadPoolExecutor cannot stop a function that is already running.
        # Exit the isolated worker after persisting completed results so timed-
        # out threads and their CUDA context are released immediately.
        os._exit(0)


def executeParallelThreaded(func, args, max_workers, batch_timeout, log_level=2):
    """
    Execute functions in parallel using threads with a batch deadline.
    Each batch runs in an isolated child process that is terminated afterward
    to fully free GPU memory.

    Designed for JAX/GPU workloads:
    - Threads within each batch share a single CUDA context (no per-worker overhead)
    - Process isolation between batches guarantees GPU memory cleanup
    - Uses 'spawn' start method (safe for CUDA on Linux/WSL)

    Example: 1000 jobs, max_workers=10, batch_timeout=10s
        -> 100 batches x 10s = ~1000s total

    Args:
        func: Function to call. Must be importable (top-level). Called as func(*arg).
        args: List of argument tuples. Must be picklable.
        max_workers: Number of concurrent threads and batch size.
        batch_timeout: Shared deadline in seconds for the concurrently-started
            jobs in one batch. Running Python threads cannot be forcefully
            cancelled, so the outer process is terminated after this deadline.
        log_level: 0=silent, 1=errors only, 2=verbose.

    Returns:
        List of results in the same order as args. Failed/timed-out jobs are None.
    """
    ctx = multiprocessing.get_context('spawn')

    results = [None] * len(args)
    total_jobs = len(args)
    completed_count = 0
    timed_out_count = 0
    failed_count = 0
    num_batches = math.ceil(total_jobs / max_workers)
    # Grace period for the child to save results after thread timeout
    grace_seconds = 5

    for batch_start in range(0, total_jobs, max_workers):
        batch_end = min(batch_start + max_workers, total_jobs)
        batch_num = batch_start // max_workers + 1
        batch_args = args[batch_start:batch_end]

        result_file = tempfile.mktemp(suffix='.pkl')

        proc = ctx.Process(
            target=_batch_worker,
            args=(func, batch_args, max_workers, batch_timeout, result_file),
        )
        proc.start()
        proc.join(timeout=batch_timeout + grace_seconds)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()

        # Read results from the child process
        batch_completed = 0
        if os.path.exists(result_file):
            try:
                with open(result_file, 'rb') as f:
                    data = dill.load(f)
                for i, r in enumerate(data['results']):
                    if r is not None:
                        results[batch_start + i] = r
                        batch_completed += 1
                for idx, err_str in data['errors']:
                    failed_count += 1
                    if log_level >= 1:
                        print(f"Warning: Job {batch_start + idx} failed: {err_str}")
                for idx in data['timed_out']:
                    timed_out_count += 1
                    if log_level >= 2:
                        print(
                            f"Warning: Job {batch_start + idx} timed out "
                            f"after {batch_timeout}s"
                        )
                completed_count += batch_completed
            except Exception as e:
                if log_level >= 1:
                    print(f"Warning: Failed to read batch {batch_num} results: {e}")
            finally:
                os.unlink(result_file)
        else:
            # Child was killed before saving — entire batch timed out
            timed_out_count += len(batch_args)
            if log_level >= 2:
                print(
                    f"Warning: Batch {batch_num} killed before saving results"
                )

        if log_level >= 2:
            print(
                f"Batch {batch_num}/{num_batches}: "
                f"{batch_completed}/{len(batch_args)} completed"
            )

    if log_level >= 1:
        print(
            f"\nSummary: {completed_count} completed, {timed_out_count} timed out, "
            f"{failed_count} failed out of {total_jobs} total jobs"
        )

    return results
