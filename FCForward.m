function [ output_data ] = FCForward(input, layer)
input_data = input.data;
W = layer.W;
b = layer.b;
output_data = bsxfun(@plus, W*input_data, b);
end

