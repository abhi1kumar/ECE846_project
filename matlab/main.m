options     = optimoptions('fmincon','display','none');
num_weights = 500; % num of weighted sum
num_seeds   = 10;  % num of repeats to be done


fprintf("=============== FON 2 =====================\n")
d   = 2;
x_init_matrix = FON(num_weights, num_seeds, d, options);

fprintf("=============== FON 3 =====================\n")
d   = 3;
x_init_matrix = FON(num_weights, num_seeds, d, options);

fprintf("=============== FON 4 =====================\n")
d   = 4;
x_init_matrix = FON(num_weights, num_seeds, d, options);

fprintf("=============== FON 8 =====================\n")
d   = 8;
x_init_matrix = FON(num_weights, num_seeds, d, options);

fprintf("=============== FON 16 =====================\n")
d   = 16;
x_init_matrix = FON(num_weights, num_seeds, d, options);

fprintf("=============== FON 32 =====================\n")
d   = 32;
x_init_matrix = FON(num_weights, num_seeds, d, options);

fprintf("=============== FON 64 =====================\n")
d   = 64;
x_init_matrix = FON(num_weights, num_seeds, d, options);



fprintf("=============== ZDT1 2 =====================\n")
x_init_matrix = ZDT1(num_weights, num_seeds, 2, options);

fprintf("=============== ZDT1 3 =====================\n")
x_init_matrix = ZDT1(num_weights, num_seeds, 3, options);

fprintf("=============== ZDT1 4 =====================\n")
x_init_matrix = ZDT1(num_weights, num_seeds, 4, options);

fprintf("=============== ZDT1 8 =====================\n")
x_init_matrix = ZDT1(num_weights, num_seeds, 8, options);

fprintf("=============== ZDT1 16 =====================\n")
x_init_matrix = ZDT1(num_weights, num_seeds, 16, options);

fprintf("=============== ZDT1 30 =====================\n")
x_init_matrix = ZDT1(num_weights, num_seeds, 30, options);




fprintf("=============== ZDT4 2 =====================\n")
x_init_matrix = ZDT4(num_weights, num_seeds, 2, options);

fprintf("=============== ZDT4 3 =====================\n")
x_init_matrix = ZDT4(num_weights, num_seeds, 3, options);

fprintf("=============== ZDT4 4 =====================\n")
x_init_matrix = ZDT4(num_weights, num_seeds, 4, options);

fprintf("=============== ZDT4 8 =====================\n")
x_init_matrix = ZDT4(num_weights, num_seeds, 8, options);

fprintf("=============== ZDT4 10 =====================\n")
x_init_matrix = ZDT4(num_weights, num_seeds, 10, options);

fprintf("=============== ZDT4 16 =====================\n")
x_init_matrix = ZDT4(num_weights, num_seeds, 16, options);
