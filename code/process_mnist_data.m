clear

training_data_raw  =  loadMNISTImages('../data/input/raw/mnist/train-images-idx3-ubyte');
training_label_raw = loadMNISTLabels('../data/input/raw/mnist/train-labels-idx1-ubyte');
testing_data_raw = loadMNISTImages('../data/input/raw/mnist/t10k-images-idx3-ubyte');
testing_label_raw =loadMNISTLabels('../data/input/raw/mnist/t10k-labels-idx1-ubyte');

save( '../data/input/mnist_training_data_raw', 'training_data_raw');
save( '../data/input/mnist_training_label_raw', 'training_label_raw');
save( '../data/input/mnist_testing_data_raw', 'testing_data_raw');
save( '../data/input/mnist_testing_label_raw', 'testing_label_raw');

