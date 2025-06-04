%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File    : BFGS.m                                                        %
%                                                                         %
% Author  : Tobias Holicki                                                %
% Version : 01                                                            %
% Date    : 31.01.2022                                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This function implements the BFGS method.
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
%            disp - Display progress if this field is nonzero.
%              B0 - Initial approximation of the inverse Hessian.
% ----- Output ---------------------------------------------------------
%        x - Approximate minimizer.
%       fx - Approximate minimal value.
%
function [x, fx] = BFGS(fg, x0, opt)
%% Some default options
% Maximum number of iterations
opt = ifnotisfield(opt, 'max_it', 50);

% Maximum number of iterations for the stepsize computations
opt = ifnotisfield(opt, 'max_it_stepsize', 100);

% Terminate if norm of gradient is below this value
opt = ifnotisfield(opt, 'term_norm_grad', 1e-5);

% Terminate if function value is below this value
opt = ifnotisfield(opt, 'term_fun_val', -Inf);

% Parameters for Powell-Wolfe stepsize rule
opt = ifnotisfield(opt, 'gamma', 0.01);
opt = ifnotisfield(opt, 'eta', 0.9);

% Override display function if no display is desired
if ~isfield(opt, 'disp') || opt.disp == 0
    dis = @(x) 0; 
else
    dis = @(x) disp(x);
end

% Approximation of the inverse Hessian
if ~isfield(opt, 'B0') % Heuristic for initial choice of B
    x       = x0;
    [fx, g] = fg(x); % Get function value and gradient
    d       = -g;    % Descent direction

    % Determine stepsize
    s = stepsize(x, fx, g, d);

    % Updates
    x           = x + s * d; % New iterate
    [fx, g_new] = fg(x);     % Get new function value and gradient
    td          = s * d;     % Part of the approximation 
    y           = g_new - g; % Another part
    g           = g_new;     % Update gradient

    % Heuristic
    B = eye(length(x)) * (y' * td) / norm(y)^2;
    te  = (td - B * y) * td';
    te2 = (td - B * y)' * y;
    te3 = td' * y;
    B   = B + (te + te') / te3 - te2 * (td * td') / te3^2;
else
    % Initializations
    x       = x0;
    B       = opt.B0;
    [fx, g] = fg(x);
end

% Main loop
for k = 1 : opt.max_it 
    % Descent direction
    d = - B * g;

    % Determine stepsize
    s = stepsize(x, fx, g, d);

    % Updates
    x           = x + s * d;      % New iterate
    [fx, g_new] = fg(x);          % Get new function value and gradient
    normgrad    = norm(g_new, 2); % Norm of new gradient

    % Terminate potentially earlier
    if normgrad < opt.term_norm_grad || fx < opt.term_fun_val
        break
    end
    dis([num2str(k), ': ', 'normgrad = ', num2str(normgrad), ', ', ...
                                 'fx = ', num2str(fx)]);

    % Update B
    td  = s * d;     
    y   = g_new - g; 
    te  = (td - B * y) * td';
    te2 = (td - B * y)' * y;
    te3 = td' * y;
    B   = B + (te + te') / te3 - te2 * (td * td') / te3^2;

    % Terminate potentially earlier
    if te3 < opt.term_norm_grad
        break;
    end

    % Update gradient
    g = g_new;     
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
