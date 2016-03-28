function [input_data_diff, layer_train] = ReLUBackward(input, output, layer, layer_train)
output_data_diff = output.data_diff;
input_data = input.data;
input_data_diff = double(input_data > 0) .* output_data_diff;
layer_train = [];
end

