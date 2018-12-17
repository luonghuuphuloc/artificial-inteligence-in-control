%% Fitness function
function y = myFitness(x)
y = -(4*x.^4 - 5*x.^3 + exp(-2.*x) - 7.*sin(x) - 3.*cos(x));

end