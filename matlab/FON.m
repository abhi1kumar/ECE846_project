function x_init_matrix = FON(num_init, N, d, options)

    % Create weight matrix
    weight_matrix = zeros(num_init, d);
    delta = 1.0/(num_init-1);
    for k = 1:num_init
        weight_matrix(k,:) = [(k-1)*delta, 1.0 - (k-1)*delta];
    end
    
    A   = [];
    b   = [];
    Aeq = [];
    beq = [];
    
    lb  = -1.0* ones(1,d);
    ub  =  1.0* ones(1,d);
    x_init_matrix = zeros(num_init, d);
    
    % Get the population member one from each point.
    for k = 1:num_init
        weight = weight_matrix(k,:);
    
        objective = @(x)FON_objective(x,weight,d);
        constraint= @constraint_empty;
        
        f_min = 10000;
        for i = 1:N
            x0 = init(lb, ub, (i-1)/N);
            [x, f] = fmincon(objective,x0,A,b,Aeq,beq,lb,ub,constraint, options);
            if f < f_min
                f_min = f;
                x_min = x;
            end
        end
        x_init_matrix(k,:) = x_min;
    end
    
    save_filename = strcat('FON_', int2str(d), '.mat');
    save(save_filename, 'x_init_matrix')
end