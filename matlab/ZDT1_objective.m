function f = ZDT1_objective(x, weight, d)
    f_1 = x(1);
    g   = 1 + 9 *(sum(x) - x(1))/(d-1);
    h   = 1 - sqrt(f_1/g);
    f_2 = g *h;
    f   = weight(1) * f_1 + weight(2) * f_2;
end
