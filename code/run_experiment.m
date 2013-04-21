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

% test to run options: '_NT';  '_ST'; '_DT'; 'ADT';
options.exp_to_run = [  '_NT'; '_ST'; ];

% set the model parameters
parameters.lambda_stepsize = 0.1
parameters.lambda_over_lambdamax = 0:parameters.lambda_stepsize:1;
%parameters = train_model(options, data);


NT_filename = sprintf([options.working_path options.data_name '_' 'NT' '_' num2str(options.random_seed)...
            '_' num2str(options.training_size) '_' num2str(options.testing_size) '_' ...
            num2str(parameters.lambda_stepsize) '.mat']);
ST_filename = sprintf([options.working_path options.data_name '_' 'ST' '_' num2str(options.random_seed)...
            '_' num2str(options.training_size) '_' num2str(options.testing_size) '_' ...
            num2str(parameters.lambda_stepsize) '.mat']);


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
oneSided = 1;


% set lambda
lambda_max = max(training_data'*testing_sample);
lambda = parameters.lambda_over_lambdamax*lambda_max;

%TODO design a better framework to run all the experiments

% solve lasso without screening
if strmatch('_NT', options.exp_to_run)
  fprintf('noscreening\n')
  if exist(NT_filename, 'file')
    fprintf('load from cache\n')
    load(NT_filename);
  else
    solve_wo_screening_time = nan(length(lambda),1); 
    t=0;
    for l=lambda
      fprintf('%f',l)
      t=t+1;
      tic_start = tic; 
      wout = l1ls_featuresign (training_data, testing_sample, l);
      solve_wo_screening_time(t) = toc(tic_start);
    end
  end
end


% run ST
if strmatch('_ST', options.exp_to_run) 
  fprintf('ST\n')
  if exist(ST_filename, 'file')
    fprintf('load from cache\n')
    load(ST_filename);
  else
    rejection_ST= nan(length(lambda),1);
    solve_w_screening_time_ST = nan(length(lambda),1); 
    t=0;
    for l=lambda
      t=t+1;
      [reject_tmp   solve_w_screening_time_ST(t)]=lasso_screening_ST(training_data,testing_sample,l,verbose,vt_feasible, oneSided);
      rejection_ST(t) = sum(reject_tmp)/length(reject_tmp);
      tic_start = tic; 
      wout = l1ls_featuresign (training_data(:,~reject_tmp), testing_sample, l);
      solve_w_screening_time_ST(t) = toc(tic_start) + solve_w_screening_time_ST(t);
    end
  end
end

% run DT
if strmatch('_DT', options.exp_to_run) 
  fprintf('DT\n')
  rejection_DT= nan(length(lambda),1);
  solve_w_screening_time_DT = nan(length(lambda),1); 
  t=0;
  for l=lambda
    t=t+1;
    [reject_tmp   solve_w_screening_time_DT(t)]=lasso_screening_DT(training_data,testing_sample,l,verbose,vt_feasible, oneSided,0);
    rejection_DT(t) = sum(reject_tmp)/length(reject_tmp);
    tic_start = tic; 
    wout = l1ls_featuresign (training_data(:,~reject_tmp), testing_sample, l);
    solve_w_screening_time_DT(t) = toc(tic_start)+solve_w_screening_time_DT(t);
  end
end

% run ADT
if strmatch('ADT', options.exp_to_run) 
  fprintf('ADT\n')
  rejection_ADT= nan(length(lambda),1);
  solve_w_screening_time_ADT = nan(length(lambda),1); 
  t=0;
  for l=lambda
    t=t+1;
    [reject_tmp   solve_w_screening_time_ADT(t)]=lasso_screening_DT(training_data,testing_sample,l,verbose,vt_feasible, oneSided,0);
    rejection_ADT(t) = sum(reject_tmp)/length(reject_tmp);
    tic_start = tic; 
    wout = l1ls_featuresign (training_data(:,~reject_tmp), testing_sample, l, 0);
    solve_w_screening_time_ADT(t) = toc(tic_start) + solve_w_screening_time_ADT(t);
  end
end

% save data to file to eliminate keep reruning the same stuff
% data_name = [options.data_name]_[testname]_[randomseed]_[options.training_size]_
%             [options.testing_size]_[labmda_stepsize]
if exist('solve_wo_screening_time','var')
save(sprintf([options.working_path options.data_name '_' 'NT' '_' num2str(options.random_seed)...
            '_' num2str(options.training_size) '_' num2str(options.testing_size) '_' ...
            num2str(parameters.lambda_stepsize) '.mat']), 'solve_wo_screening_time');
end       

if exist('solve_w_screening_time_ST','var') & exist('rejection_ST','var')
save(sprintf([options.working_path options.data_name '_' 'ST' '_' num2str(options.random_seed)...
            '_' num2str(options.training_size) '_' num2str(options.testing_size) '_' ...
            num2str(parameters.lambda_stepsize) '.mat']), 'solve_w_screening_time_ST', 'rejection_ST' );
end

if exist('solve_w_screening_time_DT','var') & exist('rejection_DT','var')
save(sprintf([options.working_path options.data_name '_' 'DT' '_' num2str(options.random_seed)...
            '_' num2str(options.training_size) '_' num2str(options.testing_size) '_' ...
            num2str(parameters.lambda_stepsize) '.mat']), 'solve_w_screening_time_DT', 'rejection_DT' );
end

if exist('solve_w_screening_time_ADT','var') & exist('rejection_ADT','var')
save(sprintf([options.working_path options.data_name '_' 'ADT' '_' num2str(options.random_seed)...
            '_' num2str(options.training_size) '_' num2str(options.testing_size) '_' ...
            num2str(parameters.lambda_stepsize) '.mat']), 'solve_w_screening_time_ADT', 'rejection_ADT' );
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
plot(lambda, rejection_ST,'*b-') 
%plot(lambda, rejection_DT,'or-')
%plot(lambda, rejection_ADT,'xg-') 
legend('ST','DT','ADT')
xlabel('lambda')
ylabel('rejection rate')
title('screening rejection rate')
mkdir(options.output_path , [date '_' options.data_name] )
saveas(gcf, [ options.output_path [date '_' options.data_name] '/' 'rejection' ], 'tiff')

%TODO speed up calculation
figure
hold on
plot(lambda, solve_wo_screening_time./solve_w_screening_time_ST,'*b-') 
%plot(lambda, solve_wo_screening_time./solve_w_screening_time_DT,'or-')
%plot(lambda, solve_wo_screening_time./solve_w_screening_time_ADT,'xg-') 
legend('ST','DT','ADT')
xlabel('lambda')
ylabel('rejection rate')
title('screening speedup')
mkdir(options.output_path , [date '_' options.data_name] )
saveas(gcf, [ options.output_path [date '_' options.data_name] '/' 'speedup' ], 'tiff')

