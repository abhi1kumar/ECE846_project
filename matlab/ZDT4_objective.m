function f = ZDT4_objective(x, weight, d)
    f_1 = x(1);
    g   = 1 + 10*(d-1) + sum(power(x,2)) - x(1)*x(1) - sum(10 * cos(4 * pi * x)) + 10 * cos(4 * pi * x(1));
    h   = 1 - sqrt(f_1/g);
    f_2 = g *h;
    f   = weight(1) * f_1 + weight(2) * f_2;
end
