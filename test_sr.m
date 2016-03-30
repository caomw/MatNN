% net1 more layers, net2 weight decay = 0.0005, net 3 weight decay = 0.005
for net_id = 2:3
    for im_id = 1:10
        load(['net' num2str(net_id) '.mat']);
        load(['testData/test_' num2str(im_id) '.mat']);
        data = [test_ori; test_filtered];
        data = double(data);
        data = bsxfun(@rdivide, bsxfun(@minus, data, data_mean), data_std + eps);
        
        net.feature_train{2}.input.data = data;
        for layer_id = 2:length(net.layers)-1
            net.feature_train{layer_id+1}.input.data = net.layers{layer_id}.forward(net.feature_train{layer_id}.input, net.layers{layer_id});
        end
        
        pre = net.feature_train{layer_id+1}.input.data;
        pre = bsxfun(@plus, bsxfun(@times, pre, label_std), label_mean);
        save(['testData/net-' num2str(net_id) '-im-' num2str(im_id) '.mat'], 'pre');
    end
end
% a = reshape(pre, 9,9,[]);
% b = reshape(label, 9,9,[]);
% for i = 1:64
%     figure(1), subplot(8,8,i); imshow(a(:,:,i+258),[]);
% end
% for i = 1:64
%     figure(2), subplot(8,8,i); imshow(b(:,:,i+258),[]);
% end
% 
% c = reshape(test_ori, 9,9,[]);
% c = c + a;
% for i = 1:64
%     figure(3), subplot(8,8,i); imshow(c(:,:,i+1000));
% end
% d = reshape(test_ori, 9,9,[]);
% d = d + b;
% for i = 1:64
%     figure(4), subplot(8,8,i); imshow(d(:,:,i+1000));
% end
% e = reshape(test_ori, 9,9,[]);
% for i = 1:64
%     figure(5), subplot(8,8,i); imshow(e(:,:,i+1000));
% end


