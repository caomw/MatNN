function [output_data] = GMMEuclideanForward(input, layer)
input_data = input.data;
label = input.label;
posterior = input.posterior;
num_gmm = size(posterior, 1);

[dim, batch_num] = size(input_data);
feature_dim = dim / num_gmm;
% input_data = reshape(input_data, [], num_gmm);

output_data = input_data-repmat(label, num_gmm, 1);
output_data = output_data.^2;
output_data = reshape(output_data, feature_dim, []);
output_data = sum(output_data, 1);
output_data = 0.5 * 1/batch_num * output_data * posterior(:);
% posterior = repmat(posterior(:), 1, feature_dim);
% posterior = reshape(posterior', dim, batch_num);
% output_data = output_data.*posterior;
% output_data = 0.5 * 1/batch_num * sum(output_data(:));

% j=1;
% for i = 1:64
%     figure(1); subplot(8,8,i); imshow(reshape(input_data([1:81] + j*81, i), 9, 9),[]);
% end

for i = 1:64
    a = reshape(input_data(:,i), feature_dim, [])*posterior(:,i);
    figure(2); subplot(8,8,i); imshow(reshape(a, 9, 9));
end

b = reshape(label, 9,9,64);
for i = 1:64
    figure(3); subplot(8,8,i); imshow(b(:,:,i));
end
