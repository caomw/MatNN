function [input, layer_train] = EuclideanBackward(input, output_diff, layer, layer_train)
input_data = input.data;
label = input.label;
sz = size(input_data, 2);
loss = (input_data-label).^2;
loss = 0.5 * 1/sz * sum(loss(:));
input.data_diff = 1/sz * (input_data - label);
% if loss < 15
%     input_data_diff = input_data_diff * 0;
% end

layer_train = [];
end


