% net1 more layers, net2 weight decay = 0.0005, net 3 weight decay = 0.005
for net_id = 1
    for im_id = 1:6
        load(['net-gmm-' num2str(net_id) '-wotraindata.mat']);
        load(['test/test_' num2str(im_id) '.mat']);
        data = [test_ori; test_filtered];
        data = double(data);
        data = bsxfun(@rdivide, bsxfun(@minus, data, data_mean), data_std + eps);
        posteriors = posteriors';
        net.feature_train{2}.input.data = data;
        
        for layer_id = 2:length(net.layers)-1
            net.feature_train{layer_id+1}.input.data = net.layers{layer_id}.forward(net.feature_train{layer_id}.input, net.layers{layer_id});
        end
        
        pre = net.feature_train{layer_id+1}.input.data;
        [dim, sample_num] = size(pre);
        gmm_num = size(posteriors, 1);
        feature_dim = dim / gmm_num;
        pre_high_resolution = nan(feature_dim, sample_num);
        for sample_id = 1:sample_num
            pre_high = reshape(pre(:, sample_id), feature_dim, gmm_num) * posteriors(:, sample_id);
            pre_high = bsxfun(@plus, bsxfun(@times, pre_high, label_std), label_mean);
            pre_high_resolution(:, sample_id) = pre_high;
        end
        save(['test/net-' num2str(net_id) '-im-' num2str(im_id) '.mat'], 'pre_high_resolution');
    end
end

a = reshape(pre_high_resolution, 9,9,[]);
b = reshape(label, 9,9,[]);
for i = 1:64
    figure(1), subplot(8,8,i); imshow(a(:,:,i+258),[]);
end
for i = 1:64
    figure(2), subplot(8,8,i); imshow(b(:,:,i+258),[]);
end

c = reshape(test_ori, 9,9,[]);
c = c + a;
for i = 1:64
    figure(3), subplot(8,8,i); imshow(c(:,:,i+1000));
end
d = reshape(test_ori, 9,9,[]);
d = d + b;
for i = 1:64
    figure(4), subplot(8,8,i); imshow(d(:,:,i+1000));
end
e = reshape(test_ori, 9,9,[]);
for i = 1:64
    figure(5), subplot(8,8,i); imshow(e(:,:,i+1000));
end


