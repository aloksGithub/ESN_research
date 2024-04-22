from scipy.special import comb
import tracemalloc

def measure_memory_usage(func, *args, **kwargs):
    """
    Measure the peak memory usage of a function.

    Parameters:
        func (callable): The function to measure.
        *args: Arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        float: The peak memory usage in MB during the function's execution.
    """
    tracemalloc.start()
    func(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_memory_MB = peak / (1024 * 1024)
    return peak_memory_MB

def estimate_lms_memory(input_dim, output_dim):
    return output_dim * (0.00003302808602650960234375 * input_dim + 0.0012229283650716145833)

def estimate_rls_memory(input_dim):
    return 0.000022 * input_dim**2

def estimate_ridge_memory(output_dim, input_shape):
    """
    Estimate memory used by Ridge readout in MB.

    Parameters:
        output_dim (int): The number of output neurons.
        input_shape (int[]): Two element list containing the number of data points and the number of features for each data point.

    Returns:
        float: Estimated memory usage in megabytes.
    """
    # Calculate array sizes for initialize function
    weight_size = input_shape[1] * output_dim
    bias_size = output_dim

    # Calculate array sizes for initialize_buffers function
    buffer_size = input_shape[1]**2 + input_shape[1] * output_dim

    # Calculate array sizes for partial backward function
    X_size = input_shape[0] * (input_shape[1] + 1)
    Y_size = input_shape[0] * output_dim
    xxt_size = (input_shape[1] + 1)**2
    yxt_size = output_dim * (input_shape[1] + 1)
    partial_backward_size = X_size + Y_size + xxt_size + yxt_size

    # Calculate array sizes for backward function
    ridgeId_size = 2 * (input_shape[1] + 1)**2
    solve_size = (input_shape[1] + 1)**2
    backward_size = ridgeId_size + solve_size
    
    # Combine size of arrays that can exist at the same time in memory
    total_size = weight_size + bias_size + buffer_size + max(backward_size, partial_backward_size)
    return 8 * total_size / (1024**2)

def estimate_nvar_memory(delay, order, input_shape):
    """
    Estimate memory used by NVAR node in MB.

    Parameters:
        delay (int): Delay parameter of NVAR node.
        order (int): Order of non-linear terms.
        input_shape (int[]): Two element list containing the number of data points and the number of features for each data point.
        dtype (np.dtype): Data type of the arrays.

    Returns:
        float: Estimated memory usage in megabytes.
    """
    float_size = 8
    effective_linear_dim = delay * input_shape[1]
    store_memory = effective_linear_dim * float_size * input_shape[0]

    # Calculate memory for output
    nonlinear_dim = comb(effective_linear_dim + order - 1, order, exact=True)
    total_output_dim = effective_linear_dim + nonlinear_dim
    output_memory = total_output_dim * float_size * input_shape[0]

    # Memory for index arrays
    idx_memory = comb(effective_linear_dim + order - 1, order, exact=True) * order * float_size

    # Convert total memory from bytes to megabytes and adjust by a factor to align with empirical data
    total_memory_mb = (store_memory + output_memory + idx_memory) / (1024 * 1024)

    return total_memory_mb

def estimate_reservoir_memory(units, input_shape):
    """
    Estimate the memory used during the reservoir computation in MB, including batch size.

    Parameters:
        units (int): Number of units in the reservoir.
        input_shape (int[]): Two element list containing the number of data points and the number of features for each data point.
        dtype (np.dtype): Data type of the arrays (default is np.float32).

    Returns:
        float: Estimated memory usage in megabytes (MB).
    """
    float_size = 8
    memory = 0

    # Memory for W (units x units), Win (units x input_dim), and optional Wfb (units x feedback_dim)
    memory += units * units * float_size  # W
    memory += units * input_shape[1] * float_size  # Win

    # Memory for bias, input u, and state r
    memory += units * float_size  # bias
    memory += input_shape[0] * input_shape[1] * float_size  # u, multiplied by batch_size
    memory += input_shape[0] * units * float_size  # r, assuming r is retained for each sample

    # Assuming noise vectors are the same size as input and state vectors
    memory += input_shape[0] * input_shape[1] * float_size  # noise for u, multiplied by batch_size
    memory += input_shape[0] * units * float_size  # noise for r (state noise), multiplied by batch_size

    # Convert memory from bytes to megabytes
    memory_mb = memory / (1024 * 1024)

    return memory_mb

def estimateMemory(model, numInputs):
    model.initialize()
    memories = []
    for node in model.nodes:
        name = node.name.split("-")[0]
        if name=="Concat" or name=="Input":
            continue
        if name=="LMS":
            memory = estimate_lms_memory(node.input_dim, node.output_dim)
        elif name=="RLS":
            memory = estimate_rls_memory(node.input_dim)
        elif name=="Ridge":
            memory = estimate_ridge_memory(node.output_dim, [numInputs, node.input_dim])
        elif name=="NVAR":
            memory = estimate_nvar_memory(node.delay, node.order, [numInputs, node.input_dim])
        elif name=="Reservoir" or name=="IPReservoir":
            memory = estimate_reservoir_memory(node.units, [numInputs, node.input_dim])
        memories.append(memory)
    return max(memories)