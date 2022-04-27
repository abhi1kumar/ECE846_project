function f = FON_objective(x, weight, d)
    k = 1.0/sqrt(d);
    f = weight(1) * (1 - exp (sum(- power(x - k, 2) )) ) + weight(2) * (1 - exp (sum(- power(x + k, 2) )) );
end
