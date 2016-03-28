function [net] = TrainNet(net)

max_iter = net.net_param.train_param.max_iter;
gamma = net.net_param.train_param.gamma;
lr = net.net_param.train_param.lr;
mm = net.net_param.train_param.mm;
step_size = net.net_param.train_param.step_size;
test_interval = net.net_param.test_param.test_interval;
display_interval = net.net_param.train_param.display_interval;
% batch_num = net.net_param.train_param.train_num/net.net_param.train_param.batch_size;
for iter = 1: max_iter
    %% input:
    data_batch = net.layers{1}.forward(net.feature_train{1}.input, net.layers{1});
    batch_id = net.layers{1}.batch_id + 1;
    if batch_id > net.layers{1}.batch_num
        batch_id = 1;
    end
    net.layers{1}.batch_id = batch_id;
    net.feature_train{2}.input.data= data_batch.data; % assign a batch of training sample to the input data of the first trainable layer.
    net.feature_train{end-1}.input.label = data_batch.label; % assign the corresponding labels to the input label of the loss layer. 
    %% forward trainable layers
    for layer_id = 2:length(net.layers)
        net.feature_train{layer_id+1}.input.data = net.layers{layer_id}.forward(net.feature_train{layer_id}.input, net.layers{layer_id});
    end
    
    %% backward trainable layers
    for layer_id = length(net.layers):-1:2
        [net.feature_train{layer_id}.input.data_diff, net.layers_train{layer_id}] ...
            = net.layers{layer_id}.backward(net.feature_train{layer_id}.input, net.feature_train{layer_id+1}.input, net.layers{layer_id}, net.layers_train{layer_id});
    end
    %% update trainable layers and ignore the loss layer
    for layer_id = length(net.layers)-1:-1:2
        [net.layers{layer_id}, net.layers_train{layer_id}] ...
            = net.layers{layer_id}.update(net.layers{layer_id}, net.layers_train{layer_id}, mm, lr);
    end
    %% lower learning rate
    if mod(iter, step_size) == 0
        lr = lr * gamma;
    end
    %% display
    if mod(iter, display_interval) == 0
        fprintf('Training #Iter = %d, Loss: %04f \n', iter, net.feature_train{length(net.layers)+1}.input.data);
    end
    %% test the network
    if mod(iter, test_interval) == 0
        [net.layers{1}.test_param.batch_id, accuracy] = TestNet(net);
        fprintf('Test #Iter = %d, Accuracy: %03f \n', iter, accuracy);
    end
        
end
end

