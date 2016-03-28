function [ output ] = InputForward(input, layer)
batch_sz = layer.batch_sz;
batch_id = layer.batch_id;
output.data = input.data(:, [1:batch_sz] + (batch_id-1) * batch_sz);
output.label = input.label(:, [1:batch_sz] + (batch_id-1) * batch_sz);
end

