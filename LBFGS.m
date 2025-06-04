%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File    : LBFGS.m                                                        %
%                                                                         %
% Author  : Tobias Holicki                                                %
% Version : 01                                                            %
% Date    : 31.01.2022                                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This function implements the LBFGS method (Algorithm 7.5) from 
% Nocedal and Wright, 2006. 
%
% ----- Input ---------------------------------------------------------- 
%       fg - If [f, g] = fg(x) then f is the function to be minimized
%            evaluated at x and g is the corresponding gradient.
%       x0 - Starting point of the algorithm
%      opt - Struct with (possible) fields
%          max_it - Maximum number of iterations.
% max_it_stepsize - Maximum number of iterations for the loops in the
%                   stepsize rule.
%  term_norm_grad - Terminate if the gradient is below this value.
%    term_fun_val - Terminate if the function value is below this value.
%           gamma - Parameter in the Powell-Wolfe stepsize rule.
%             eta - Parameter in the Powell-Wolfe stepsize rule.
%          memory - Number of saved directions.
%            disp - Display progress if this field is nonzero.
% ----- Output ---------------------------------------------------------
%        x - Approximate minimizer.
%       fx - Approximate minimal value.
%
function [x, fx] = LBFGS(fg, x0, opt)
%% Some default options
% Maximum number of iterations
opt = ifnotisfield(opt, 'max_it', 300);

% Maximum number of iterations for the stepsize computations
opt = ifnotisfield(opt, 'max_it_stepsize', 100);

% Terminate if norm of gradient is below this value
opt = ifnotisfield(opt, 'term_norm_grad', 1e-5);

% Terminate if function value is below this value
opt = ifnotisfield(opt, 'term_fun_val', -Inf);

% Parameters for Powell-Wolfe stepsize rule
opt = ifnotisfield(opt, 'gamma', 0.01);
opt = ifnotisfield(opt, 'eta', 0.9);

% Used memory
opt = ifnotisfield(opt, 'memory', 10);

% Override display function if no display is desired
if ~isfield(opt, 'disp') || opt.disp == 0
    dis = @(x) 0; 
else
    dis = @(x) disp(x);
end

% Initialization
a   = zeros(1, opt.memory);
b   = zeros(1, opt.memory);
rho = zeros(1, opt.memory);
s   = zeros(length(x0), opt.memory);
y   = zeros(length(x0), opt.memory);

x       = x0;
[fx, g] = fg(x); % Get function value and gradient

% First step for more initializations...
d  = -g;
al = stepsize(x, fx, g, d);
[fx, g_new] = fg(x + al * d);

s(:, 1) = al * d; % = x_new - x
y(:, 1) = g_new - g;
rho(1)  = 1/ (y(:, 1)' * s(:, 1));


% Main loop
for k = 1 : opt.max_it 
    % Descent direction via two-loop recursion
    d = twoloop_recursion(k, g_new, s, y, rho);

    % Determine stepsize
    al = stepsize(x, fx, g, d);

    % Updates
    g           = g_new;          % Previous gradient
    x           = x + al * d;      % New iterate
    [fx, g_new] = fg(x);          % Get new function value and gradient
    normgrad    = norm(g_new, 2); % Norm of new gradient

    % Terminate potentially earlier
    if normgrad < opt.term_norm_grad || fx < opt.term_fun_val
        break
    end
    dis([num2str(k), ': ', 'normgrad = ', num2str(normgrad), ', ', ...
                                 'fx = ', num2str(fx)]);

    % Update memorized directions
    if k >= opt.memory
        s   = [s(:, 2:end), al * d];
        y   = [y(:, 2:end), g_new - g];
        rho = [rho(2:end), 1 / (y(:, end)' * s(:, end))];
    else
        s(:, k+1) = al * d;
        y(:, k+1) = g_new - g;
        rho(k+1)  = 1 / (y(:, k+1)' * s(:, k+1)); 
    end
end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % function: twoloop_recursion
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Determines the new descent direction by Algorithm 7.4 in Nocedal and 
    % Wright 2006.
    function d = twoloop_recursion(k, q, s, y, rho)
        m  = min(k, opt.memory); % For the initial phase
        % Estimated size of Hessian along current search direction
        ga = (s(:, m)' * y(:, m)) / norm(y(:, m))^2;    
        for i = m : -1: 1
           a(i) = rho(i) * (s(:, i)' * q);
           q = q - a(i) * y(:, i);
        end
        r = q * ga;
        for i = 1 : m
            b(i) = rho(i) * (y(:, i)' * r);
            r = r + s(:, i) * (a(i) - b(i));
        end
        d = -r;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % function: stepsize
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Determines stepsize based on Powell-Wolfe rule
    function s = stepsize(x, fx, g, d)
        ip  = opt.gamma * g' * d; % Time saver
        ip2 = opt.eta * g' * d; % 

        % First loop (breaks if Armijo condition is satisfied)
        sm = 2;
        for i = 1 : opt.max_it_stepsize
           sm     = sm / 2;         % Decrease stepsize
           fxnew = fg(x + sm * d); % Update function value
           if fxnew <= fx + sm * ip 
               break;
           end
        end
        
        % Second loop (breaks if Armijo condition is not satisfied)
        sp = sm;
        for i = 1 : opt.max_it_stepsize
            sp     = sp * 2;         % Increase stepsize
            fxnew = fg(x + sp * d); % Update function value
            if fxnew > fx + sp * ip
                break;
            end
        end
        
        % Final loop (this is a bisection)
        [~, gnew] = fg(x + sm * d);
        for i = 1 : opt.max_it_stepsize
            if gnew' * d >= ip2
                break;
            end
            s0     = (sp + sm) / 2;  % Mean
            xnew  = x + s0 * d;      % Update iterate
            fx_new = fg(x + s0 * d); % Update function value
            if fx_new <= fx + s0 * ip
                sm        = s0;
                [~, gnew] = fg(xnew); % Update gradient
            else
                sp = s0;
            end
            
        end
        % Final stepsize
        s = sm;
    end

end
