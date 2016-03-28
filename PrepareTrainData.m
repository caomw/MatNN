function [data_set] = PrepareTrainData
data = loadMNISTImages('/home/lijun/Research/Code/caffe/data/mnist/train-images-idx3-ubyte');
label = loadMNISTLabels('/home/lijun/Research/Code/caffe/data/mnist/train-labels-idx1-ubyte');

data_mean= mean(data, 2);
data_var = std(data,0, 2);
data = bsxfun(@rdivide, bsxfun(@minus, data, data_mean), data_var+eps);
categories = length(unique(label));
sample_num = length(label);
ind = sub2ind([categories, sample_num], (label+1)', 1:sample_num);
label = zeros(categories, sample_num);
label(ind) = 1;
data_set.data = data;
data_set.label = label;
data_set.mean = data_mean;
data_set.std = data_var;
end

