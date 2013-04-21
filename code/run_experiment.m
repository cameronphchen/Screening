% experiment for various lasso screening
% by Cameron P.H. Chen @ Princeton

clear

% set the algorithm options
fprintf('loading options\n')
options.data_name = 'mnist';
options.input_path = '../data/input/'; 
options.training_dataset = 'train-images-idx3-ubyte';
options.training_label = 'train-labels-idx1-ubyte';
options.testing_dataset = 't10k-images-idx3-ubyte';
options.testing_label = 't10k-labels-idx1-ubyte';
options.working_path = '../data/working/' ; 
options.output_path = '../data/output/' ; 
options.random_seed = 99;
options.training_size = 5000;
options.testing_size = 1000;

% test to run options: 'NT';  'ST'; 'DT'; 'ADT';
options.exp_to_run = {  'NT'; 'ST'; 'DT'; 'ADT'};
screening_function_handle = {@rand,@lasso_screening_ST, @lasso_screening_DT, @lasso_screening_ADT};
% set the model parameters
parameters.lambda_stepsize = 0.1;
parameters.lambda_over_lambdamax = 0:parameters.lambda_stepsize:1;
%parameters = train_model(options, data);

filenames = cell(length(options.exp_to_run),1);

for i = 1:length(options.exp_to_run)
filenames{i} = sprintf([options.working_path options.data_name '_' options.exp_to_run{i} '_' num2str(options.random_seed)...
            '_' num2str(options.training_size) '_' num2str(options.testing_size) '_' ...
            num2str(parameters.lambda_stepsize) '.mat']);
end

% load the data 
fprintf('loading data\n')
training_data_raw  =  loadMNISTImages([options.input_path options.training_dataset]);
training_label_raw = loadMNISTLabels([options.input_path options.training_label]);
testing_data_raw = loadMNISTImages([options.input_path options.testing_dataset]);
testing_label_raw =loadMNISTLabels([options.input_path options.testing_label]);

% select the training and testing data with uniform distribution accross differnet label
assert (options.training_size < size(training_data_raw,2),...
        'number of codeword to select from training data > num of codeword in raw data\n')
assert (options.testing_size < size(training_data_raw,2),...
        'number of codeword to select from testing data > num of codeword in raw data\n')

training_data  = nan(size(training_data_raw,1),options.training_size);
training_label = nan(options.training_size,1);
testing_data   =nan(size(testing_data_raw,1),options.testing_size);
testing_label  = nan(options.testing_size,1);

for i=1:10

  tmp_train_idc=randsample(find(training_label_raw==(i-1)),options.training_size/10);
  tmp_test_idc=randsample(find(testing_label_raw==(i-1)),options.testing_size/10);

  training_data(:,((i-1)*(options.training_size/10)+1):i*(options.training_size/10))=...
        training_data_raw(:,tmp_train_idc);

  training_label(((i-1)*(options.training_size/10)+1):i*(options.training_size/10))=...
        training_label_raw(tmp_train_idc);

  testing_data(:,((i-1)*(options.testing_size/10)+1):i*(options.testing_size/10))=...
        testing_data_raw(:,tmp_test_idc);

  testing_label(((i-1)*(options.testing_size/10)+1):i*(options.testing_size/10))=...
        testing_label_raw(tmp_test_idc);
end 

% data normalization
training_data = training_data./...
                sqrt((ones(size(training_data,1),1)*sum((training_data.^2),1)));
testing_data = testing_data./...
                sqrt((ones(size(testing_data,1),1)*sum((testing_data.^2),1)));

assert(norm(sum(sum(training_data.^2))-size(training_data,2))<0.01,'training_data normailization failed')
assert(norm(sum(sum(testing_data.^2))-size(testing_data,2))<0.01,'testing_data normailization failed')

% select screening data
rng(options.random_seed);
testing_sample = testing_data(:,ceil(rand*size(testing_data,2)));

% random select a certain amount of data and



% set screening options
verbose = 1;
vt_feasible = [];
oneSided = 0;


% set lambda
lambda_max = max(training_data'*testing_sample);
lambda = parameters.lambda_over_lambdamax*lambda_max;

%TODO design a better framework to run all the experiments

rejection=nan(length(lambda),length(options.exp_to_run));
solve_w_screening_time=nan(length(lambda),length(options.exp_to_run));

% solve lasso without screening

if isequal(options.exp_to_run{1},'NT')
  fprintf('noscreening\n')
  if exist(filenames{1}, 'file')
    fprintf('load from cache\n')
    load(filenames{1});
    solve_w_screening_time(:,1) = tmp_screening_time;
  else
    t=0;
    for l=lambda
      fprintf('%f',l)
      t=t+1;
      tic_start = tic; 
      wout = l1ls_featuresign (training_data, testing_sample, l);
      solve_w_screening_time(t,1) = toc(tic_start);
    end
  end
end

% run screening

for i = 2:length(options.exp_to_run)
  fprintf([options.exp_to_run{i} '\n'])
  if exist(filenames{i}, 'file')
    fprintf('load from cache\n')
    load(filenames{i});
    solve_w_screening_time(:,i) = tmp_screening_time;
    rejection(:,i)=tmp_rejection_rate;
  else
    t=0;
    for l=lambda
      t=t+1;
      [reject_tmp  solve_w_screening_time(t,i)]=screening_function_handle{i}...
                  (training_data,testing_sample,l,verbose,vt_feasible, oneSided);
      rejection(t,i) = sum(reject_tmp)/length(reject_tmp);
      tic_start = tic; 
      wout = l1ls_featuresign (training_data(:,~reject_tmp), testing_sample, l);
      solve_w_screening_time(t,i) = toc(tic_start) + solve_w_screening_time(t,i);
    end
  end
end

% caching

for i = 1:length(options.exp_to_run)
  tmp_screening_time = solve_w_screening_time(:,i);
  assert(sum(isnan(tmp_screening_time))==0, sprintf('nan in %s screening time',options.exp_to_run{i}));
  if isequal(options.exp_to_run{i},'NT') 
    save(filenames{i}, 'tmp_screening_time');
  else
    tmp_rejection_rate = rejection(:,i);
    save(filenames{i}, 'tmp_screening_time', 'tmp_rejection_rate' );
  end
end

% run THT 
% run IDT
% run THT with MP selected two codeword
% run THT with OMP selected two codeword
% run IDT with MP selected codewords
% run IDT with OMP selected codewords

% TODO design a better way to select the line that we want to plot

% plot results
figure
hold on

for i = 2:length(options.exp_to_run)
  plot(lambda, rejection(:,i),'*b-') 
end
legend('ST','DT','ADT')
xlabel('lambda')
ylabel('rejection rate')
title('screening rejection rate')
mkdir(options.output_path , [date '_' options.data_name] )
saveas(gcf, [ options.output_path [date '_' options.data_name] '/' 'rejection' ], 'tiff')

%TODO speed up calculation
figure
hold on
for i = 2:length(options.exp_to_run)
  plot(lambda, solve_w_screening_time(:,1)./solve_w_screening_time(:,i),'*b-') 
end
legend('ST','DT','ADT')
xlabel('lambda')
ylabel('rejection rate')
title('screening speedup')
mkdir(options.output_path , [date '_' options.data_name] )
saveas(gcf, [ options.output_path [date '_' options.data_name] '/' 'speedup' ], 'tiff')

