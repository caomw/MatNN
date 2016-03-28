function [input_data_diff, layer_train] = EuclideanBackward(input, output_diff, layer, layer_train)
input_data = input.data;
label = input.label;
sz = size(input_data, 2);
input_data_diff = 1/sz * (input_data - label);
layer_train = [];
end


