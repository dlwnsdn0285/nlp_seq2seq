# seq2seq translation model using LSTM cells

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import datasets
import tqdm
import evaluate

dataset = datasets.load_dataset()
