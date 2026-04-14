"""Shared seeding helper for experiments."""
import os
import random
import numpy as np


def set_global_seed(seed):
    """Seed all libraries that can introduce randomness."""
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
