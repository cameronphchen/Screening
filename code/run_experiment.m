% experiment for various lasso screening
% by Cameron P.H. Chen @ Princeton

clear


% set the algorithm options
fprintf('loading options\n')
options.data_name = 'mnist';
options.time = clock
options.time = [date '-' num2str(options.time(4)) num2str(options.time(5))]
options.input_path = '../data/input/'; 
options.training_dataset = 'train-images-idx3-ubyte';
options.training_label = 'train-labels-idx1-ubyte';
options.testing_dataset = 't10k-images-idx3-ubyte';
options.testing_label = 't10k-labels-idx1-ubyte';
options.working_path = '../data/working/' ; 
options.output_path = '../data/output/' ; 
options.random_seed = 99;
options.training_size = 500;
options.testing_size = 100;
options.num_iter = 10;
% test to run options: 'NT';  'ST'; 'DT'; 'ADT';
options.exp_to_run = {  'NT'; 'ST'; 'DT'; 'ADT'; 'ADT-MP'; 'ADT-OMP';};

screening_function_handle = {@rand,@lasso_screening_ST, @lasso_screening_DT, @lasso_screening_ADT,...
                             @lasso_screening_ADT_MP ,@lasso_screening_ADT_OMP};
% set the model parameters
parameters.lambda_stepsize = 0.05;
parameters.lambda_over_lambdamax = 0:parameters.lambda_stepsize:1;
%parameters = train_model(options, data);

filenames = cell(length(options.exp_to_run),1);

for i = 1:length(options.exp_to_run)
filenames{i} = sprintf([options.working_path options.data_name '_' options.exp_to_run{i} '_' num2str(options.random_seed)...
            '_' num2str(options.training_size) '_' num2str(options.testing_size) '_' num2str(options.num_iter) '_'...
            num2str(parameters.lambda_stepsize)  '.mat']);
end

% create diary to save log
delete diary.txt
diary diary.txt
diary on;

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
testing_sample = testing_data(:,ceil(rand(options.num_iter,1)*size(testing_data,2)));

% set screening options
verbose = 1;
vt_feasible = [];
oneSided = 0;

%TODO design a better framework to run all the experiments

rejection=zeros(length(parameters.lambda_over_lambdamax),length(options.exp_to_run))
solve_w_screening_time=zeros(length(parameters.lambda_over_lambdamax),length(options.exp_to_run))

% run screening

for i = 1:length(options.exp_to_run)
  fprintf([options.exp_to_run{i} '\n'])
  if exist(filenames{i}, 'file')
    fprintf('load from cache\n')
    load(filenames{i});
    solve_w_screening_time(:,i) = tmp_screening_time;
    if ~isequal(options.exp_to_run{i},'NT')
      rejection(:,i)=tmp_rejection_rate
    end
  else
    for j=1:options.num_iter

      % set lambda
      lambda_max = max(training_data'*testing_sample(:,j));
      lambda = parameters.lambda_over_lambdamax*lambda_max;

      for t=1:length(parameters.lambda_over_lambdamax)
        reject_tmp = zeros(options.training_size,1);
        if ~isequal(options.exp_to_run{i},'NT')
          [reject_tmp  solve_w_screening_time(t,i)]=screening_function_handle{i}...
                  (training_data,testing_sample(:,j),lambda(t),verbose,vt_feasible, oneSided);
          rejection(t,i) = sum(reject_tmp)/length(reject_tmp) + rejection(t,i)
        end
        tic_start = tic; 
        wout = l1ls_featuresign (training_data(:,~reject_tmp), testing_sample(:,j), lambda(t));
        solve_w_screening_time(t,i) = toc(tic_start) + solve_w_screening_time(t,i)
      end
    end
    solve_w_screening_time(:,i) = solve_w_screening_time(:,i)./options.num_iter
    rejection(:,i) = rejection(:,i)./options.num_iter 
  end
end



% caching

for i = 1:length(options.exp_to_run)
  tmp_screening_time = solve_w_screening_time(:,i);
  assert(sum(tmp_screening_time)~=0, sprintf('zero for all %s screening time',options.exp_to_run{i}));
  if isequal(options.exp_to_run{i},'NT') 
    save(filenames{i}, 'tmp_screening_time');
  else
    tmp_rejection_rate = rejection(:,i);
    save(filenames{i}, 'tmp_screening_time', 'tmp_rejection_rate' );
  end
end

% run IDT
% run IDT with MP selected codewords
% run IDT with OMP selected codewords

% TODO design a better way to select the line that we want to plot

line_color  = ['c','g','b','r','m','y','k','w'];
line_marker = ['.','o','*','+','x','s','d'];
line_style  = ['-','--',':','-.'];

outputfolder =  [options.time '_' options.data_name '_' num2str(options.random_seed)...
            '_' num2str(options.training_size) '_' num2str(options.testing_size) '_' num2str(options.num_iter) '_'...
            num2str(parameters.lambda_stepsize)]

mkdir(options.output_path , outputfolder )


close all
% plot results
figure
hold on
for i = 2:length(options.exp_to_run)
  plot(parameters.lambda_over_lambdamax, rejection(:,i),[line_color(i) line_marker(i) line_style(1)]) 
end
legend(options.exp_to_run(2:end))
xlabel('lambda/lambda\_max')
ylabel('rejection rate')
title('screening rejection rate')
saveas(gcf, [ options.output_path outputfolder  '/' 'rejection' ], 'tiff')

% plot speed up
figure
hold on
for i = 2:length(options.exp_to_run)
  plot(parameters.lambda_over_lambdamax, solve_w_screening_time(:,1)./solve_w_screening_time(:,i),...
        [line_color(i) line_marker(i) line_style(1)]) 
end
legend(options.exp_to_run(2:end))
xlabel('lambda/lambda\_max')
ylabel('rejection rate')
title('screening speedup')
saveas(gcf, [ options.output_path outputfolder  '/' 'speedup' ], 'tiff')

% plot speed up vs results
figure
hold on
for i = 2:length(options.exp_to_run)
  scatter(rejection(:,i), solve_w_screening_time(:,1)./solve_w_screening_time(:,i),[line_color(i) line_marker(i)]) 
end
legend(options.exp_to_run(2:end))
xlabel('rejection rate')
ylabel('speed up')
title('screening speedup vs rejection')
saveas(gcf,  [ options.output_path outputfolder  '/' 'speedup_vs_rejection'  ], 'tiff')
save( [ options.output_path outputfolder  '/' 'options_and_parameter'], 'options', 'parameters');

diary off;
