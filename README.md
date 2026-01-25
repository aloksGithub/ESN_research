# ESN Research

A PhD research project to improve the predictive capabilities of Echo State Networks using Architecture Search.

## Requirements

- **Python**: 3.10 or higher (tested with Python 3.12)
- **OS**: Linux (including WSL2), macOS, or Windows
- **GPU** (optional): NVIDIA GPU with CUDA support for acceleration

## Installation

### Basic Installation (CPU)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### GPU Installation (NVIDIA CUDA)

For GPU acceleration, install JAX with CUDA support after the basic installation:

```bash
# Uninstall CPU-only JAX
pip uninstall jax jaxlib -y

# Install JAX with CUDA 12 support
pip install --upgrade "jax[cuda12]"
```

Verify GPU is detected:
```bash
python -c "import jax; print(jax.devices())"
# Should output: [CudaDevice(id=0)]
```

### Verify Installation

Run the test script to verify everything works:
```bash
python test_jax_migration.py
```

## Project Structure

```
ESN_research/
├── experiments/         # Experiment scripts (GA, BO, ESNAS)
├── src/                 # Source code
│   ├── algorithms/      # ESN optimization algorithms
│   ├── nodes/           # Custom reservoir nodes
│   └── ...
├── scripts/             # Analysis and visualization scripts
├── data/                # Dataset files
├── results/             # Experiment results
└── requirements.txt     # Python dependencies
```

## Running Experiments

```bash
# Genetic Algorithm experiments
python experiments/ga.py

# Bayesian Optimization experiments
python experiments/bo.py

# ESNAS (combined GA + BO) experiments
python experiments/esnas.py
```

## Datasets

The project includes experiments on several time series datasets:
- Lorenz attractor
- Mackey-Glass series
- Santa Fe laser
- Sunspots
- Water demand
- DDE (Delay Differential Equation)

## License

See [LICENSE](LICENSE) file.
