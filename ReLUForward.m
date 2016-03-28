function [output_data] = ReLUForward(input, layer)
input_data = input.data;
output_data = double(input_data>0) .* input_data;
end

