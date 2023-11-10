import sys
import torch
import matplotlib.pyplot as plt

from gan_mnist_cond_generator import Generator
from gan_mnist_cond_utils import generate_random_seed

G = Generator()
G.load_model()

_, axarr = plt.subplots(2,3, figsize=(16,8))

# define the label
label = int(sys.argv[1])
label_tensor = torch.zeros((10))
label_tensor[label] = 1

# generate images
for i in range(2):
    for j in range(3):
        seed = generate_random_seed(100)
        output = G.forward(seed, label_tensor)
        img = output.detach().numpy().reshape(28,28)
        axarr[i,j].imshow(img, interpolation='none', cmap='Blues')
        pass
    pass

plt.show()