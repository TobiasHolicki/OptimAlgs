%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File    : Globalized_L_BFGS.m                                           %
%                                                                         %
% Author  : Tobias Holicki                                                %
% Version : 01                                                            %
% Date    : 10.07.2020                                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This function realizes the globalized L-BFGS method.
%
function [z, fz] = Globalized_L_BFGS(f, gradf, z0, opt)
%% Some default options
% Maximum number of iterations
if ~isfield(opt, 'max_it')
    opt.max_it = 50;
end

% Maximum number of iterations for the stepsize computations
if ~isfield(opt, 'max_it_stepsize')
    opt.max_it_stepsize = 100;
end

% Terminate if norm of gradient is below this value
if ~isfield(opt, 'term_norm_grad')
    opt.term_norm_grad = 1e-5;
end

% Terminate if function value is below this value
if ~isfield(opt, 'term_fun_val')
    opt.term_fun_val = -Inf;
end

% Armijo-Parameter
if ~isfield(opt, 'gamma')
    opt.gamma = 0.01;
end
if ~isfield(opt, 'eta')
    opt.eta = 0.9;
end

% Memory
if ~isfield(opt, 'memory')
    opt.memory = 10;
end

% Override display function if no display is desired
if ~isfield(opt, 'disp') || opt.disp == 0
    dis = @(x) 0; 
else
    dis = @(x) disp(x);
end



% Initialization
z      = z0;
fz     = f(z0);
gradfk = gradf(z0);

a   = zeros(1, opt.memory);
b   = zeros(1, opt.memory);
rho = zeros(1, opt.memory);
s   = zeros(length(z0), opt.memory);
y   = zeros(length(z0), opt.memory);

% First step
z_new     = z - gradfk * PW_stepsize(fz, z, -gradfk, gradfk, f, gradf, opt);
gradf_new = gradf(z_new);
fz        = f(z_new);

s(:, 1) = z_new - z;
y(:, 1) = gradf_new - gradfk;
rho(1)  = 1 / (y(:, 1)' * s(:, 1));

% Main loop
for k = 1 : opt.max_it   
    %% Descent direction  
    m = min(k, opt.memory);  % For the initial phase
    gak = s(:, m)' * y(:, m) / norm(y(:, m))^2;                  
    
    q = gradf_new;
    for i = m : -1: 1
       a(i) = rho(i) * s(:, i)' * q;
       q = q - a(i) * y(:, i);
    end
    r = q * gak;
    for i = 1 : m
        b(i) = rho(i) * y(:, i)' * r;
        r = r + s(:, i) * (a(i) - b(i));
    end
    dk = -r;
    
    %% Updates
    z = z_new;
    z_new = z_new + dk * PW_stepsize(fz, z_new, dk, gradf_new, f, gradf, opt);
    
    gradfk    = gradf_new;
    gradf_new = gradf(z_new);
    
    
    fz        = f(z_new);           % New function value
    normgrad  = norm(gradf_new, 2); % Norm of new gradient
    
    % Terminate potentially earlier
    if normgrad < opt.term_norm_grad || fz < opt.term_fun_val
        break
    end
    dis([num2str(k), ': ', 'normgrad = ', num2str(normgrad), ', ', ...
                                 'fz = ', num2str(fz)]);
    
    if k >= opt.memory
        s   = [s(:, 2:end), z_new - z];
        y   = [y(:, 2:end), gradf_new - gradfk];
        rho = [rho(2:end), 1 / (y(:, end)' * s(:, end))];
    else
        s(:, k+1) = z_new - z;
        y(:, k+1) = gradf_new - gradfk;
        rho(k+1)  = 1 / (y(:, k+1)' * s(:, k+1)); 
    end
end

end



function sk = PW_stepsize(fz, z, dk, gradfk, f, gradf, opt)
    %% Determine stepsize via Powell-Wolfe rule

    % Abbreviation
    ip = opt.gamma * gradfk' * dk;

    % First loop (breaks if Armijo condition is satisfied)
    sm = 2;
    for i = 1 : opt.max_it_stepsize
       sm     = sm / 2;         % Decrease stepsize
       fz_new = f(z + sm * dk); % Update function value
       if fz_new <= fz + sm * ip 
           break;
       end
    end

    % Second loop (breaks if Armijo condition is not satisfied)
    sp = sm;
    for i = 1 : opt.max_it_stepsize
        sp     = sp * 2;         % Increase stepsize
        fz_new = f(z + sp * dk); % Update function value
        if fz_new > fz + sp * ip
            break;
        end
    end

    % Final loop (this is a bisection)
    gradf_new = gradf(z(:, end) + sm * dk);
    for i = 1 : opt.max_it_stepsize
        if gradf_new' * dk >= opt.eta * ip / opt.gamma
            break;
        end
        s0     = (sp + sm) / 2;  % Mean
        z_new  = z + s0 * dk;    % Update iterate
        fz_new = f(z_new);       % Update function value
        if fz_new <= fz + s0 * ip
            sm        = s0;
            gradf_new = gradf(z_new); % Update gradient
        else
            sp = s0;
        end

    end

    % Final Schrittweite
    sk = sm; 

end

