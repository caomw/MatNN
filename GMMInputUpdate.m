function [posterior] = GMMInputUpdate(layer, posterior, posterior_diff)
fr = layer.posterior_forget_rate;
batch_sz = layer.batch_sz;
batch_num = layer.batch_num;
batch_id = layer.batch_id;
batch_id = batch_id - 1;
if batch_id <= 0
    batch_id = batch_num;
end

posterior(:, [1:batch_sz] + (batch_id-1) * batch_sz) = fr * posterior(:, [1:batch_sz] + (batch_id-1) * batch_sz) ...
    + (1-fr) * posterior_diff;

end

