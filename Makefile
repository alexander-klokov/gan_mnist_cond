train:
	python3 gan_mnist_cond_train.py

test:
	python3 gan_mnist_cond_test.py $(label)
