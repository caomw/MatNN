function [layer, layer_train] = FCUpdate(layer, layer_train, moment, lr)
layer_train.W_hist = moment * layer_train.W_hist - lr * layer_train.W_diff;
layer_train.b_hist = moment * layer_train.b_hist - lr * layer_train.b_diff;
layer.W = layer.W + layer_train.W_hist;
layer.b = layer.b + layer_train.b_hist;
end

