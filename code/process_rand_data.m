training_data_raw=rand(28,10000);
testing_data_raw = rand(28,1000);
training_label_raw = rand(10000,1);
testing_label_raw = rand(1000,1);

training_label_raw = (training_label_raw > 0.5);
testing_label_raw = (testing_label_raw > 0.5);


save( '../data/input/rand_training_data_raw', 'training_data_raw');
save( '../data/input/rand_training_label_raw', 'training_label_raw');
save( '../data/input/rand_testing_data_raw', 'testing_data_raw');
save( '../data/input/rand_testing_label_raw', 'testing_label_raw');
