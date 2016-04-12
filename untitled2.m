% x = 0:0.1:500;
% l = repmat(log(a), length(x), 1);
% b = sum(exp(l .* repmat(x', 1, 100)), 2);
% 
% y = m.^(x') ./ b;
% figure();plot(x,y);
x = (0:0.01:400)' + 0;
a = posterior_diff(1:100) - max(posterior_diff(:));
a = a - min(a);
a = repmat(a, length(x), 1);
a = a .* repmat(x, 1, 100);
p = exp(-a);
y = max(p, [], 2)./ sum(p, 2);
sig = 1./x;
figure(); plot(sig,y)