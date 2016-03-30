function [input_data_diff, layer_train] = EuclideanBackward(input, output_diff, layer, layer_train)
input_data = input.data;
label = input.label;
posterior = input.posterior;
num_gmm = size(posterior, 1);
[dim, batch_sz] = size(input_data);
feature_dim = dim / num_gmm;
% loss = (input_data-label).^2;
% loss = 0.5 * 1/sz * sum(loss(:));
input_data_diff = 1/batch_sz * (input_data - repmat(label, num_gmm, 1));
posterior = repmat(posterior(:), 1, feature_dim);
posterior = reshape(posterior', dim, batch_sz);
input_data_diff = input_data_diff .* posterior;
% if loss < 15
%     input_data_diff = input_data_diff * 0;
% end

layer_train = [];
end


