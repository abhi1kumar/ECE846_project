options  = optimoptions('fmincon','display','none');
num_init = 500;
N   = 10;


fprintf("=============== FON 2 =====================\n")
d   = 2;
x_init_matrix = FON(num_init, N, d, options);