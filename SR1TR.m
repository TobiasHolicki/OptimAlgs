%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File    : SR1TR.m                                                       %
%                                                                         %
% Author  : Tobias Holicki                                                %
% Version : 01                                                            %
% Date    : 31.01.2022                                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This function implements the SR1 Trust-Region method (Algorihm 6.2) in
% Nocedal and Wright.
%
% ----- Input ---------------------------------------------------------- 
%       fg - If [f, g] = fg(x) then f is the function to be minimized
%            evaluated at x and g is the corresponding gradient.
%       x0 - Starting point of the algorithm
%      opt - Struct with (possible) fields
%          max_it - Maximum number of iterations.
%  term_norm_grad - Terminate if the gradient is below this value.
%    term_fun_val - Terminate if the function value is below this value.
%            eta1 - Parameter in the TR radius update.
%            eta2 - Parameter in the TR radius update.
%            eta3 - Parameter in the TR radius update.
%          Delmin - Minimal TR radius after a successful step.
%               r - Parameter that determines when to skip an update of the
%                   Hessian approximation.
%            disp - Display progress if this field is nonzero.
%              B0 - Initial approximation of the Hessian.
%            Del0 - Initial TR radius.
% ----- Output ---------------------------------------------------------
%        x - Approximate minimizer.
%       fx - Approximate minimal value.
%
function [x, fx] = SR1TR(fg, x0, opt)
%% Some default options
% Maximum number of iterations
opt = ifnotisfield(opt, 'max_it', 250);

% Terminate if norm of gradient is below this value
opt = ifnotisfield(opt, 'term_norm_grad', 1e-5);

% Terminate if function value is below this value
opt = ifnotisfield(opt, 'term_fun_val', -Inf);

% Parameters for updating the TR radius
opt = ifnotisfield(opt, 'eta1', 0.1);
opt = ifnotisfield(opt, 'eta2', 0.75);
opt = ifnotisfield(opt, 'eta3', 0.8);
opt = ifnotisfield(opt, 'Delmin', 0.1); 

% Parameter for updating the approximation of the hessian
opt = ifnotisfield(opt, 'r', 1e-8);

% Override display function if no display is desired
if ~isfield(opt, 'disp') || opt.disp == 0
    dis = @(x) 0; 
else
    dis = @(x) disp(x);
end

% Initial approxiation of the Hessian
n = length(x0);
opt = ifnotisfield(opt, 'B0', eye(n) + ones(n)/n);
B = opt.B0;

% Initial TR radius
opt = ifnotisfield(opt, 'Del0', 10);
Del = opt.Del0;

%% Main Part

% Initializations
x       = x0;
[fx, g] = fg(x);

% Main loop
for k = 1 : opt.max_it 
    % Solve the trust-region subproblem
    s = TRsubproblem(g, B, Del);

    % Potential new function value and gradient
    [fx_new, g_new] = fg(x + s);
    
    % Terminate potentially earlier
    normgrad = norm(g_new);
    if normgrad < opt.term_norm_grad || fx < opt.term_fun_val
        x  = x + s;
        fx = fx_new;
        break
    end
    dis([num2str(k), ': ', 'normgrad = ', num2str(normgrad), ', ', ...
                                 'fx = ', num2str(fx), ', ', ...
                              'Delta = ', num2str(Del)]);
    
    y    = g_new - g;
    ared = fx - fx_new; % Actual reduction
    pred = -(g' * s + 0.5 * s' * B * s); % Predicted reduction
    
    % Potentially update iterate as well as function value and gradient
    if ared / pred > opt.eta1
        x  = x + s; 
        fx = fx_new;
        g  = g_new;
    end % else do not update
    
    % Potentially update trust region
    Del = updateTRradius(Del, ared / pred, norm(s));
    
    % Potentially update approximation of Hessian
    w = y - B * s; % Potential time saver
    if abs(s' * w) >= opt.r * norm(s) * norm(w)
        B = B + (w * w') / (w' * s); % Update formula        
    else
        % Skip the update
    end 
end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Function: updateTRradius
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Update heuristic for the radius of the trust region with a small
    % modification for the SR1 method.
    function Del = updateTRradius(Del, rho, ns)
        if rho <= opt.eta1 % Shrink TR radius since prediction was bad
            Del = 0.5 * Del;  
        elseif rho > opt.eta1 && rho <= opt.eta2 % Keep TR radius
            Del = max(opt.Delmin, Del);
        elseif rho > opt.eta2 % Maybe increase TR radius
            if ns > opt.eta3 * Del 
                Del = max(opt.Delmin, 2*Del);
            end
        end
    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Function: TRsubproblem
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Determines the solution of the two dimensional TR subproblem
    function s = TRsubproblem(g, B, Del)
        % Build two-dimensional subspace 
        % B should not be the identity and should be invertible
        w1 = g;
        w2 = -B \ g;    
    
        % Apply Gram-Schmitt for orthonormalization
        v1 = w1 / norm(w1, 2);
        v2 = w2 - (v1' * w2) * v1;
        v2 = v2 / norm(v2, 2);
    
        % Set up two the corresponding two dimensional problem
        %            min_s g' * s + 0.5 * s' * H * s  subject to |s| <= Del
        V = [v1, v2];
        g = V' * g;
        B = V' * B * V;
    
        % If H is positive definite, then the following s is the solution 
        % of the unconstrained problem
        s = - B \ g;
        e = eigs(B, 1, 'smallestabs'); % s could be a saddle point
        if norm(s) > Del || e < 0 
            % In this case, we have solve the KKT conditions
            %      g + H*s + la *s = 0  and   0.5*(s'*s - Del^2) = 0
            % instead. The first equation gives  
            %      s = -(H + la I)^{-1} * g
            % and the second one (after multiplying with det(H + la I)^2)
            %      0 = g' * adj(H + la I)^2 * g - Del^2 * det(H + la I)^2.
            % This is a polynomial of degree 4 in la and we can find the 
            % zeros thereof explicitely or numerically.
            a = B(1, 1);
            b = B(1, 2);
            c = B(2, 2);
            d = g(1);
            e = g(2);

            % The actual polynomial is expressed with coefficients co as
            %           co' * [1; la^1; la^2; la^3; la^4].
            % The concrete coefficients are computed by hand. One could
            % also make use of the symbolic toolbox which finds them as
            % well, but it takes a little longer.
            co    = zeros(5, 1); 
            co(1) = -Del^2;
            co(2) = -Del^2 * 2 * (a + c);
            co(3) = (d^2 + e^2) - Del^2 * (a^2+ c^2 + 4*a*c - 2*b^2);
            co(4) = 2*(g' *[c, -b; -b, a] * g - Del^2 * det(B) * trace(B));
            co(5) = g' * [c, -b; -b, a]^2 * g - Del^2 * det(B)^2;

            % Find zeros
            ro = roots(co);
            ro = ro(abs(imag(ro)) < 1e-8); % We only look for real ones
    
            % Find minimizer among the possibilities for s for the roots la
            qval = zeros(1, length(ro));
            ts   = zeros(2, length(ro));
            for i = 1 : length(ro)
                ts(:, i) = -(B + ro(i) *eye(2)) \ g;
                qval(i)  = g' * ts(:, i) + 0.5*ts(:, i)' * B * ts(:, i);
            end
            [~, j] = min(qval);
            s      = ts(:, j);
        end
    
        % Adjust for the original problem
        s = V * s;
    end
end
