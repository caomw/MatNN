function [input_data_diff, layer_train] = FCBackward(input, output, layer, layer_train)
W = layer.W;
b = layer.b;
weight_decay = layer.weight_decay;
input_data = input.data;
output_data_diff = output.data_diff;
W_diff = output_data_diff * input_data' + W * weight_decay;
b_diff = sum(output_data_diff, 2);
layer_train.W_diff = W_diff;
layer_train.b_diff = b_diff;
input_data_diff = W' * output_data_diff;
end

