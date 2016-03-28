function [batch_id, acc] = TestNet(net)
input_layer = net.layers{1};
batch_sz = input_layer.test_param.batch_sz;
batch_id = input_layer.test_param.batch_id;
batch_num = input_layer.test_param.batch_num;
test_set = net.feature_train{1}.test_input;
index = [1:batch_sz] + (batch_id-1) * batch_sz;
data_batch.data = test_set.data(:, index);
data_batch.label = test_set.label(:, index);
batch_id = batch_id + 1;
if batch_id > batch_num
    batch_id = 1;
end

net.feature_train{2}.input.data= data_batch.data; % assign a batch of training sample to the input data of the first trainable layer.
%% forward trainable layers, ignore loss layer
for layer_id = 2:length(net.layers)-1
    net.feature_train{layer_id+1}.input.data = net.layers{layer_id}.forward(net.feature_train{layer_id}.input, net.layers{layer_id});
end

[~, pre] = max(net.feature_train{length(net.layers)}.input.data, [], 1);
[~, label] = max(data_batch.label, [], 1);
acc = sum(double(pre == label)) / batch_sz;
end

