clear 

path='../data/input/raw/CroppedYale/data'

listing = dir(path)
numFile = size(listing,1)-3;

data = rand(32256,numFile);

training_data_raw=rand(32256,2000);
testing_data_raw = rand(32256,numFile-2000);
training_label_raw = rand(2000,1);
testing_label_raw = rand(numFile-2000,1);

training_label_raw = (training_label_raw > 0.5);
testing_label_raw = (testing_label_raw > 0.5);




for i=1:numFile
  % 3 becasue there are implicit files "." ".." ".DS_Store" 
  [pic,maxgray]= getpgmraw( [path '/' listing(i+3).name] ); 
  data(:,i) = pic(:);
end

rand_seq = randperm(numFile);

for i=1:2000
  training_data_raw(:,i) = data(:,rand_seq(i));
end

for i=2001:numFile
  testing_data_raw(:,i-2000) = data(:,rand_seq(i));
end

save( '../data/input/yalebxf_training_data_raw', 'training_data_raw');
save( '../data/input/yalebxf_training_label_raw', 'training_label_raw');
save( '../data/input/yalebxf_testing_data_raw', 'testing_data_raw');
save( '../data/input/yalebxf_testing_label_raw', 'testing_label_raw');
