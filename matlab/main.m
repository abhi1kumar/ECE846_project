options  = optimoptions('fmincon','display','none');
num_init = 500; % num of weighted sum
N   = 10;  % num of repeats to be done


fprintf("=============== FON 2 =====================\n")
d   = 2;
x_init_matrix = FON(num_init, N, d, options);

fprintf("=============== FON 3 =====================\n")
d   = 3;
x_init_matrix = FON(num_init, N, d, options);

fprintf("=============== FON 4 =====================\n")
d   = 4;
x_init_matrix = FON(num_init, N, d, options);

fprintf("=============== FON 8 =====================\n")
d   = 8;
x_init_matrix = FON(num_init, N, d, options);

fprintf("=============== FON 16 =====================\n")
d   = 16;
x_init_matrix = FON(num_init, N, d, options);

fprintf("=============== FON 32 =====================\n")
d   = 32;
x_init_matrix = FON(num_init, N, d, options);

fprintf("=============== FON 64 =====================\n")
d   = 64;
x_init_matrix = FON(num_init, N, d, options);
