SRC='src'

train:
	python3 ${SRC}/gan_mnist_cond_train.py

test:
	python3 ${SRC}/gan_mnist_cond_test.py $(label)
