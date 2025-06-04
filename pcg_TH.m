%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function : pcg.m                                                        %
%                                                                         %
% Author   : Tobias Holicki                                               %
% Version  : 01                                                           %
% Date     : 29.01.2022                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This function implements the preconditioned conjugate gradient method for
% solving the systen of linear equations Ax = b involving a symmetric and
% positive definite matrix A. Preconditioning is realized with a matrix C 
% that is a (rough) approximation of A^{-1} and which is positive definite.
% In this case we can find K with C = KK^T and can solve the equivalent
% system K^TAK (K^{-1}x) = K^Tb.
%
% ----- Input ---------------------------------------------------------- 
%        A - Involved matrix or linear operator from R^n to R^n.
%            Sometimes the latter representation can be more efficient.
%        b - Right hand side.
%      opt - Struct with (possible) fields
%            C - The preconditioner. E.g. C = diag(1./diag(A)) or
%                based on the incomplete Cholesky factorization.
%            x - Some initial guess of the solution.
%       max_it - Maximum number of iterations.
%     term_res - The algorithm terminates earlier if the residual is below
%                this value.
%         disp - Display progress if this field is nonzero.
% ----- Output ---------------------------------------------------------
%        x - Approximate solution of Ax = b.
%
function x = pcg(A, b, opt)
    %% Some default options
    % Maximum number of iterations. % Theoretically we only need length(b) 
    % iterations to find a solution, but practically this is never the case
    % due to rounding errors.
    if ~isfield(opt, 'max_it')
        opt.max_it = length(b)*3;
    end
    
    % Terminate if norm of residual is below this value
    if ~isfield(opt, 'term_res')
       opt.term_res = 1e-8;
    end
    
    % Override display function if no display is desired
    if ~isfield(opt, 'disp') || opt.disp == 0
        dis = @(x) 0; 
    else
        dis = @(x) disp(x);
    end

    % Preconditioner
    if ~isfield(opt, 'C')
        C = speye(length(b));
    else
        C = opt.C;
    end
    
    % Initial guess
    if ~isfield(opt, 'x')
        x = zeros(length(b), 1);
    else
        x = opt.x;
    end
    
    % Check if A is given as matrix or as a function/linear operator
    if isa(A, 'numeric')
       A = @(x) A * x; 
    end % Otherwise A is a linear operator from R^n to R^n
    
    %% Main algorithm
    r = b - A(x); % Initial residual
    z = C * r;
    p = z;

    % Main loop
    for k = 1 : opt.max_it
        Ap    = A(p);   % Time saver
        beta  = r' * z; % Another time saver
        alpha = beta / (p' * Ap);
        x     = x + alpha * p;  % Update approximate solution
        r     = r - alpha * Ap; % Update residual
        nores = norm(r, 2);     % Norm of residual
        if norm(r, 2) < opt.term_res % Terminate if residual is small
              break
        end
        % Print progress
        dis([num2str(k), ': Residual = ', num2str(nores)]);
        
        z    = C * r;
        beta = (r' * z) / beta;
        p    = z + beta * p;
    end
end

