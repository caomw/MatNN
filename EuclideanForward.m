function [output_data] = EuclideanForward(input, layer)
input_data = input.data;
label = input.label;
sz = size(input_data, 2);
output_data = (input_data-label).^2;
output_data = 0.5 * 1/sz * sum(output_data(:));
end

