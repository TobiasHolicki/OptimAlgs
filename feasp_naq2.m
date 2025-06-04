%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function : feasp_bfgs.m                                                 %
%                                                                         %
% Author   : Tobias Holicki                                               %
% Version  : 01                                                           %
% Date     : 01.12.2020                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% % These functions compute the function value, the gradient and both,
% respectively. All of them involve the projection Pi into the negative
% semidefinite cone. Precisely,
%            f(x) = ||F(x) - Pi(F(x))||^2_Fro,
% where F(x) is the canonical form of an LMI expression 
%            F(x) = A0 + A1*x_1 + .... + An*xn.
% Here, the projection is computed based on the eig() function.
%
% ----- Input ---------------------------------------------------------- 
%   c          - Vector defining the objective function
%   Con        - List of constraints with the same syntax as yalmip.
%   opt        - Struct
%   opt.max_it - Number of iterations for the improvements of the
%                relaxation via adapted congruence transformation
%   opt.U      - Cell of initial congruence transformation applied 
%                individually to each of the constraints. 
%                Default: Identity matrices
%   opt.disp   - Vector of output dimensions [err, mea]
%   opt.sopt   - Options for the SOCP solver coneprob
%
% ----- Output ---------------------------------------------------------
%   yvars      - Computed optimal point if the relaxed problem was feasible
%   fval       - Computed optimal value if the relaxed problem was feasible
% 
function [yvars, fval] = feasp_bfgs(Con, opt)
%% Some default options
% Maximum number of iterations
if ~isfield(opt, 'max_it')
    opt.max_it = 50;
end

% Maximum number of iterations for the stepsize computations
if ~isfield(opt, 'max_it_stepsize')
    opt.max_it_stepsize = 60;
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
    opt.gamma = 0.0001;
end
if ~isfield(opt, 'eta')
    opt.eta = 0.9;
end

% Override display function if no display is desired
if ~isfield(opt, 'disp') || opt.disp == 0
    dis = @(x) 0; 
else
    dis = @(x) disp(x);
end


%% Preparations

% Get some stuff from Yalmip
[model, rec_model, ~, int_model] = export(Con, []);

% Abbreviations
bas = -int_model.F_struc; % Standard basis
bdm  = model.prob.bardim; % Block dimensions
nc   = length(bdm);       % Number of constraints
nv   = size(bas, 2) - 1;  % Number of variables
sbdm = [0; tril(ones(nc)) * (bdm.^2)']; % Abbreviation for ofsets

% Initializations
if ~isfield(opt, 'x_init') % Decision vector
    z = zeros(nv, 1);      
else
    z = opt.x_init(:);
end
if ~isfield(opt, 'B_init') % BFGS update matrix
    Bk = eye(nv);
else
    Bk = opt.B_init;
end
[fz, gradfz] = fgradf(z); % Function value and gradient
sm_old       = 2;         % Step size guess

v = z;
mu = 0.5;

% Main loop
for k = 1 : opt.max_it   
    %% Descent direction
    dk = -Bk * gradfz;
        
    %% Determine stepsize via Powell-Wolfe rule   
    
    % Abbreviation
    ip = opt.gamma * gradfz' * dk;

    % First loop (breaks if Armijo condition is satisfied)
    sm = min(2, sm_old * 16);   % Can save time in the initial phase
    for i = 1 : opt.max_it_stepsize
       sm     = sm / 2;         % Decrease stepsize
       fz_new = f(z + sm * dk); % Update function value       
       if fz_new <= fz + sm * ip 
           break;
       end
    end
    sm_old = sm;
   
    % Second loop (breaks if Armijo condition is not satisfied)
    if i > 1
        sp = 2 * sm; % Saves one evaluation sometimes
    else
        sp = sm;
        for i = 1 : opt.max_it_stepsize
            sp     = sp * 2;         % Increase stepsize
            fz_new = f(z + sp * dk); % Update function value
            if fz_new > fz + sp * ip
                break;
            end
        end
    end

    % Final loop (this is a bisection)
    [~, gradf_new] = fgradf(z + sm * dk); 
    for i = 1 : opt.max_it_stepsize
        if abs(gradf_new' * dk) <= abs(opt.eta * ip / opt.gamma)
        %if gradf_new' * dk >= opt.eta * ip / opt.gamma
            break;
        end
        s0 = (sp + sm) / 2;  % Mean
        % Update function value und safe gradient for potential update
        [fz_new, gradf_] = fgradf(z + s0 * dk); 
        if fz_new <= fz + s0 * ip
            sm        = s0;
            gradf_new = gradf_; % Truly update gradient
        else
            sp = s0;
        end
        
    end
     % Final stepsize
     sk = sm;

    %% Updates
    v               = mu * v + sk * dk;
    z               = z + v;             % New iterate
    [fz, gradf_new] = fgradf(z);          % New function value and gradient
    normgrad        = norm(gradf_new, 2); % Norm of new gradient
    if isnan(normgrad)
        error('Something strange happened')
    end
    % Terminate potentially earlier
    if normgrad < opt.term_norm_grad || fz < opt.term_fun_val
        dis('Found local minimum within the given tolerances')
        break
    end
    
    
    
    
    
    dis([num2str(k), ': ', 'normgrad = ', num2str(normgrad), ', ', ...
                                 'fz = ', num2str(fz)]);

    % Update Bk
    tdk = sk * dk;
    yk  = gradf_new - gradfz;
    te3 = tdk' * yk;
    te  = (tdk - Bk * yk) / te3;
    te1 = te * tdk';
    te2 = (te' * yk) / te3;
    Bk  = Bk + (te1 + te1') - te2 * (tdk * tdk');
    
    % Terminate potentially earlier
    if te3 < 1e-10
        dis(['Stopped because positivity condition is only marginally', ...
              ' satisfied']);
        break;
    end   
    
    % Update gradient
    [fz, gradfz] = fgradf(z + mu * v); 
end


fval = fz;

% Yalmip variables
yvars = recover(rec_model.used_variables);

% Assigning values
assign(recover(rec_model.used_variables), z)

yvars = value(yvars);





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Functions : f, fgradf                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% These functions compute the function value, the gradient and both,
% respectively. All of them involve the projection Pi into the negative
% semidefinite cone. Precisely,
%            f(x) = ||F(x) - Pi(F(x))||^2_Fro,
% where F(x) is the canonical form of an LMI expression 
%            F(x) = A0 + A1*x_1 + .... + An*xn.
% Here, the projection is computed based on the eig() function.
% Remarks:
% - The prejection can also be computed via svd or polar decomposition.
% - It might be beneficial to save the coordinate change V for future 
%   iterations. However, when using eig() this didn't really help.
% - Maybe one can somehow exploit that F(x) is typically something like
%                A1 * X1 * B1 + ... + An * Xn * Bn + ()^T
%   with (symmetric or nonsymmetric) matrices Xi.
function [y] = f(x)
    y = 0;
    for q = 1 : nc  
        % Constraint with current values for the variables
        Fx = reshape(bas(sbdm(q)+1:sbdm(q+1), :) * [1; x], bdm(q), bdm(q));
        % Compute ingredients of projection to negative semidefinite cone

        d = eig(Fx);
        
        y = y + sum(srelu(d));
    end  
end


function [y, g] = fgradf(x)
    y = 0;
    g = zeros(nv, 1);
    for q = 1 : nc
        % Constraint with current values for the variables
        Fx = reshape(bas(sbdm(q)+1:sbdm(q+1), :) * [1; x], bdm(q), bdm(q));
        % Compute ingredients of projection to negative semidefinite cone
        [V, d] = eig(Fx, 'vector');
        
        [d1, d2] = ddxsrelu(d);
        % Projection error
        P2 = V * diag(d2) * V';
        g = g +  bas(sbdm(q)+1:sbdm(q+1), 2:end)' * P2(:);
        y = y + sum(d1); 
    end 
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Functions : srelu                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This is a smooth variant of the relu function.
%
function [x] = srelu(x)
    p = 1e-6; % A translation parameter
    % Index sets corresponding to values of interest
    ind1 = x <= -2*p;
    ind2 = x > -2*p & x < 0;
    % Apply the function
    x(ind1) = -p;
    x(ind2) = x(ind2).^2 / (4*p) + x(ind2);
end


function [x, g] = ddxsrelu(x)
    g = x;    % A copy 
    p = 1e-6; % A translation parameter
    % Index sets corresponding to values of interest
    ind1 = x <= -2*p;
    ind2 = x > -2*p & x < 0;
    ind3 = x >= 0;
    % Apply the function
    x(ind1) = -p;
    x(ind2) = x(ind2).^2 / (4*p) + x(ind2);
    % Values of the derivative
    g(ind1) = 0;
    g(ind2) = g(ind2) / (2*p) + 1;
    g(ind3) = 1;
end




end






