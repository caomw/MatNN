addpath('mnistHelper/');
load('train-SR-data-v2.mat');
data = double([data_ori; data_filtered]);
data_mean = mean(data, 2);
data_std = std(data, [], 2);
data = bsxfun(@rdivide, bsxfun(@minus, data, data_mean), data_std + eps);
label_mean = mean(label, 2);
label_std = std(label, [], 2);
label = bsxfun(@rdivide, bsxfun(@minus, label, label_mean), label_std + eps);

sample_num = size(data, 2);
test_num = round(sample_num/5);
train_num = sample_num - test_num;
data_set.data = data;
data_set.label = label;
test_set.data = [];
test_set.label = [];
% data_set.data = data(:, 1:train_num);
% data_set.label = label(:, 1:train_num);
% test_set.data = data(:, train_num+1:end);
% test_set.label = label(:, train_num+1:end);
%% %%%%%%%%%%%%%
% data_set = PrepareTrainData();
% test_set = PrepareTestData(data_set.mean, data_set.std);
net_param = InitNetParam;
% net = InitNet(net_param, data_set, test_set);
clear data_set test_set;

net = TrainNet(net);