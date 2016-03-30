function [output_data] = GMMEuclideanForward(input, layer)
input_data = input.data;
label = input.label;
posterior = input.posterior;
num_gmm = size(posterior, 1);

[dim, batch_num] = size(input_data);
feature_dim = dim / num_gmm;
% input_data = reshape(input_data, [], num_gmm);

output_data = input_data-reshape(label, num_gmm, []);
output_data = output_data.^2;
output_data = reshape(output_data, feature_dim, []);
output_data = sum(output_data, 1);
output_data = 0.5 * 1/batch_num * output_data * posterior(:);
% posterior = repmat(posterior(:), 1, feature_dim);
% posterior = reshape(posterior', dim, batch_num);
% output_data = output_data.*posterior;
% output_data = 0.5 * 1/batch_num * sum(output_data(:));
end

