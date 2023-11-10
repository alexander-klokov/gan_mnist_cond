import matplotlib.pyplot as plt

from mnist_dataset import MnistDataset

mnist_dataset_train = MnistDataset('mnist_data/mnist_train.csv')

from gan_mnist_cond_generator import Generator
from gan_mnist_cond_discriminator import Discriminator

from gan_mnist_cond_utils import generate_random_seed, generate_random_one_hot, label_true, label_false

D = Discriminator()
G = Generator()

# training
epochs = 3

for i in range(epochs):
    print('training epoch', i, 'of', epochs)

    for label, image_data_tensor, label_tensor in mnist_dataset_train:
        D.train(image_data_tensor, label_tensor, label_true)

        random_label = generate_random_one_hot(10)
        D.train(G.forward(generate_random_seed(100), random_label).detach(), random_label, label_false)
    
        random_label = generate_random_one_hot(10)
        G.train(D, generate_random_seed(100), random_label, label_true)

        pass
    pass

D.save_model()
G.save_model()

# display the progress
D.plot_progress()
G.plot_progress()

plt.show()
