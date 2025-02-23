# code to fix random seed for reproducibility

import os
import torch
import numpy as np
import random

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def my_seed_everywhere(seed: int = 42):
    random.seed(seed) # random
    np.random.seed(seed) # numpy
    os.environ["PYTHONHASHSEED"] = str(seed) # os
    # pytorch
    torch.manual_seed(seed)

    if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
      torch.use_deterministic_algorithms(True)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

'''
my_seed = 42
my_seed_everywhere(my_seed)
'''