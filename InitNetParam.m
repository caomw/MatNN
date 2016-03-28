function [net_param] = InitNetParam
train_param.lr = 0.001;
train_param.gamma = 0.1;
train_param.mm = 0.9;
train_param.weight_decay = 0.0005;
train_param.max_iter = 3 * 60000;
% train_param.train_num = 60000;
train_param.batch_size = 64;
train_param.step_size = 60000;
train_param.display_interval = 1000000;
test_param.batch_size = 1000;
test_param.test_interval = 2000;
% network architecture. Even entries indicates the feature dimension of input and output, respectively. 
architecture = {'input',[784] 'fc', [784, 256], 'relu', [256], 'fc', [256, 10], 'euclidean', [10,1]};

net_param.train_param = train_param;
net_param.test_param = test_param;
net_param.architecture = architecture;
end

