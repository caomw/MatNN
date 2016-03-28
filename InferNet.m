function [ pre ] = InferNet(net, data)
    dim = size(data, 1);
    net.feature_train{2}.input.data= data;
 %% forward trainable layers
    for layer_id = 2:length(net.layers)-1
        net.feature_train{layer_id+1}.input.data = net.layers{layer_id}.forward(net.feature_train{layer_id}.input, net.layers{layer_id});
    end
    [~, pre] = max(net.feature_train{layer_id+1}.input.data);
    figure(32); imshow(mat2gray(reshape(data, dim^0.5, dim^0.5)));
    title_str = sprintf('Pre: %d', pre-1);
    title(title_str);

end

