function y = logistic(B,x)

y = 1 ./ (1 + exp(-B(1) - B(2)*x));

end
%EOF