import os
import torch
import matplotlib.pyplot as plt

from mnist_dataset import MnistDataset

path_to_data = os.environ["DATA_CSV_MNIST"]
mnist_dataset_train = MnistDataset(path_to_data + '/mnist_train.csv')

from gan_mnist_cond_generator import Generator
from gan_mnist_cond_discriminator import Discriminator

from gan_mnist_cond_utils import generate_random_seed, generate_random_one_hot, label_true, label_false

if torch.cuda.is_available():
    print('using cuda:', torch.cuda.get_device_name(0))
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D = Discriminator()
D.to(device)

G = Generator()
G.to(device)

# training
epochs = 3

for i in range(epochs):
    print('training epoch', i, 'of', epochs)

    for label, image_data_tensor, label_tensor in mnist_dataset_train:
        D.train(image_data_tensor.to(device), label_tensor.to(device), label_true)

        random_label = generate_random_one_hot(10).to(device)
        D.train(G.forward(generate_random_seed(100).to(device), random_label).to(device).detach(), random_label, label_false)
    
        random_label = generate_random_one_hot(10).to(device)
        G.train(D, generate_random_seed(100).to(device), random_label, label_true)

        pass
    pass

D.save_model()
G.save_model()

# get some CUDA stats
print(torch.cuda.memory_summary(device, abbreviated=True))

# display the progress
D.plot_progress()
G.plot_progress()

plt.show()
