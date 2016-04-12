function [ posterior ] = ComputePosterior(net, data_set)

posterior = zeros(size(data_set.posterior));
batch_sz = 1000;
[feature_dim, num_sample] = size(data_set.label);
num_gmm = size(data_set.posterior, 1);
num_iter = floor(num_sample/batch_sz);

for iter = 1 : num_iter
    if mod(iter,100) == 0
        fprintf('Iter %d/%d \n', iter, num_iter);
    end
    net.feature_train{2}.input.data = data_set.data(:, (1:batch_sz) + (iter-1) * batch_sz);
    label = data_set.label(:, (1:batch_sz) + (iter-1) * batch_sz);
    
    %% forward trainable layers
    for layer_id = 2:length(net.layers)-1
        net.feature_train{layer_id+1}.input.data = net.layers{layer_id}.forward(net.feature_train{layer_id}.input, net.layers{layer_id});
    end
    %% compute posterior
    
    pre = net.feature_train{layer_id+1}.input.data;
    posterior_diff = (pre - repmat(label, num_gmm, 1));
    posterior_diff = reshape(posterior_diff, feature_dim, []);
    
    posterior_diff = sum(posterior_diff.^2, 1);
    % posterior_diff = posterior_diff / (max(posterior_diff(:)) + eps);
    posterior_diff = reshape(posterior_diff, num_gmm, batch_sz);
    posterior_diff = posterior_diff - repmat(min(posterior_diff, [], 1), num_gmm, 1);
    posterior_diff = exp(-posterior_diff/0.1);
    % posterior_diff = reshape(posterior_diff, num_gmm, batch_sz);
    posterior_diff = posterior_diff ./ repmat(sum(posterior_diff, 1), num_gmm, 1);
%     posterior_diff = floor(posterior_diff/0.01) * 0.01;
    posterior(:, (1:batch_sz) + (iter-1) * batch_sz) = posterior_diff ./ repmat(sum(posterior_diff, 1), num_gmm, 1);
end
%% last batch
if batch_sz + (num_iter-1) * batch_sz ~= num_sample
    num_last = num_sample - num_iter * batch_sz;
    net.feature_train{2}.input.data = data_set.data(:, end-num_last+1:end);
    label = data_set.label(:, end-num_last+1:end);
    
    %% forward trainable layers
    for layer_id = 2:length(net.layers)-1
        net.feature_train{layer_id+1}.input.data = net.layers{layer_id}.forward(net.feature_train{layer_id}.input, net.layers{layer_id});
    end
    %% compute posterior
    
    pre = net.feature_train{layer_id+1}.input.data;
    posterior_diff = (pre - repmat(label, num_gmm, 1));
    posterior_diff = reshape(posterior_diff, feature_dim, []);
    
    posterior_diff = sum(posterior_diff.^2, 1);
    % posterior_diff = posterior_diff / (max(posterior_diff(:)) + eps);
    posterior_diff = reshape(posterior_diff, num_gmm, []);
    posterior_diff = posterior_diff - repmat(min(posterior_diff, [], 1), num_gmm, 1);
    posterior_diff = exp(-posterior_diff/0.1);
    % posterior_diff = reshape(posterior_diff, num_gmm, batch_sz);
    posterior_diff = posterior_diff ./ repmat(sum(posterior_diff, 1), num_gmm, 1);
%     posterior_diff = floor(posterior_diff/0.01) * 0.01;
    posterior(:, end-num_last+1:end) = posterior_diff ./ repmat(sum(posterior_diff, 1), num_gmm, 1);
    
end

end

