function [input, layer_train] = GMMEuclideanBackward(input, output_diff, layer, layer_train)
input_data = input.data;
label = input.label;
posterior = input.posterior;
num_gmm = size(posterior, 1);
[dim, batch_sz] = size(input_data);
feature_dim = dim / num_gmm;
% loss = (input_data-label).^2;
% loss = 0.5 * 1/sz * sum(loss(:));
posterior_diff = (input_data - repmat(label, num_gmm, 1));
input_data_diff = 1/batch_sz * posterior_diff;
posterior_diff = reshape(posterior_diff, feature_dim, []);

posterior = repmat(posterior(:), 1, feature_dim);
posterior = reshape(posterior', dim, batch_sz);
input.data_diff = input_data_diff .* posterior;
posterior_diff = sum(posterior_diff.^2, 1);
% posterior_diff = posterior_diff / (max(posterior_diff(:)) + eps);
posterior_diff = reshape(posterior_diff, num_gmm, batch_sz);
posterior_diff = posterior_diff - repmat(min(posterior_diff, [], 1), num_gmm, 1);
posterior_diff = exp(-posterior_diff/0.1);
% posterior_diff = reshape(posterior_diff, num_gmm, batch_sz);
posterior_diff = posterior_diff ./ repmat(sum(posterior_diff, 1), num_gmm, 1);
posterior_diff = floor(posterior_diff/0.01) * 0.01;
input.posterior_diff = posterior_diff ./ repmat(sum(posterior_diff, 1), num_gmm, 1);
if ~(abs(sum(input.posterior_diff(:)) - batch_sz) < 0.0001)
    1;
end

layer_train = [];
end


