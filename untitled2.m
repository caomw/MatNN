x = 0:0.1:500;
l = repmat(log(a), length(x), 1);
b = sum(exp(l .* repmat(x', 1, 100)), 2);

y = m.^(x') ./ b;
figure();plot(x,y);