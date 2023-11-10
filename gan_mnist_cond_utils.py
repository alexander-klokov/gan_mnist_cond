import torch
import random

label_true = torch.FloatTensor([1.0])
label_false = torch.FloatTensor([0.0])

def generate_random_seed(size):
    random_data = torch.randn(size) # normal
    return random_data

def generate_random_one_hot(size):
    label_tensor = torch.zeros((size))
    random_idx = random.randint(0,size-1)
    label_tensor[random_idx] = 1.0
    return label_tensor