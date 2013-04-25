% sample data and normalize the dictionary
% by Cameron P.H. Chen @Princeton 
% contact: pohsuan [at] princeton [dot] edu

function [ training_data training_label testing_data testing_label ] =...
          sample_data(training_data_raw,training_label_raw,testing_data_raw,testing_label_raw,training_size,testing_size,labelNum)

training_data  = nan(size(training_data_raw,1),training_size);
training_label = nan(training_size,1);
testing_data   = nan(size(testing_data_raw,1),testing_size);
testing_label  = nan(testing_size,1);

for i=1:labelNum

  tmp_train_idc=randsample(find(training_label_raw==(i-1)),training_size/labelNum);
  tmp_test_idc=randsample(find(testing_label_raw==(i-1)),testing_size/labelNum);

  training_data(:,((i-1)*(training_size/labelNum)+1):i*(training_size/labelNum))=...
        training_data_raw(:,tmp_train_idc);

  training_label(((i-1)*(training_size/labelNum)+1):i*(training_size/labelNum))=...
        training_label_raw(tmp_train_idc);

  testing_data(:,((i-1)*(testing_size/labelNum)+1):i*(testing_size/labelNum))=...
        testing_data_raw(:,tmp_test_idc);

  testing_label(((i-1)*(testing_size/labelNum)+1):i*(testing_size/labelNum))=...
        testing_label_raw(tmp_test_idc);
end 


