%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File    : TrustRegionVoll.m                                             %
%                                                                         %
% Author  : Tobias Holicki                                                %
% Version : 01                                                            %
% Date    : 09.01.2020                                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Diese Funktion realisiert das globalisierte Newton-Verfahren 
% (Algorithmus 2.71 aus dem Skript zur Einführung in die Optimierung)
%

function [z, fz] = TrustRegionVoll(f, gradf, Hessf, z0, Del0, eta1, eta2, ...
                        gamma1, gamma2, Delmin)

% Initialisierungen und Abkürzungen zum Zeitsparen
z         = z0;
Del       = Del0;
fz        = f(z0);
gradfk    = gradf(z0);
normgradk = norm(gradfk, 2);
Hessfk    = Hessf(z0);
eps       = 1e-7;
options   = optimoptions('fmincon','Display', 'off', ...
                         'SpecifyObjectiveGradient', true, ...
                         'SpecifyConstraintGradient', true);

% Abbruch, falls der Gradient in der Norm klein genug ist 
% (||gradf|| <= 10^(-7))
while normgradk > eps
    %% Suchrichtung bestimmen 
    %normgradk
    %Del(end)
    
    % Unterraumproblem lösen mit fmincon
    fun = @(x) objective(x, fz(end), gradfk, Hessfk);
    con = @(x) constraints(x, Del(end));
    x0 = z(:, end); % 
    [dk, qkdk] = fmincon(fun, [0;0;0], [], [], [], [], [], [], con, options);
    
    % Neue Richtung und potenzielle Updates
    z_neu  = z(:, end) + dk;
    fz_neu = f(z_neu);
    
    % Quotienten berechnen
    if abs(fz(end) - qkdk) <= eps
        z  = [z, z_neu];
        fz = [fz, fz_neu];
        break;
    end   
    rho = (fz(end) - fz_neu) / (fz(end) - qkdk);
    
    % Neue Iterierte berechnen
    if rho > eta1 % Akzeptiere dk
        z_neu = z(:, end) + dk;
    else
        z_neu = z(:, end);
    end
   
    % Update Trust Region Radius
    if rho <= eta1
        Del_neu = gamma1 * Del(end); 
    elseif rho > eta1 && rho <= eta2
        Del_neu = max(Delmin, Del(end));
    elseif rho > eta2
        Del_neu = max(Delmin, gamma2*Del(end));
    end
    
    %% Ergebnisse sammeln und Abkürzungen updaten
    z         = [z, z_neu];      % Neue Iterte anhängen
    fz        = [fz, fz_neu];    % Neue Kosten anhängen
    gradfk    = gradf(z_neu);    % Gradientenauswertung festhalten   
    normgradk = norm(gradfk, 2); % Norm vom Gradienten festhalten
    Hessfk    = Hessf(z_neu);    % Hessematrixauswertung festhalten
    Del       = [Del, Del_neu];  
end

end


function [f, g] = objective(x, fz, gradfk, Hessfk)
    f = fz + gradfk' * x + 0.5 * x' * Hessfk * x;
    g = Hessfk * x + gradfk;
end


function [c, ceq, GC, GCeq] = constraints(x, Del)
    c    = x' * x - Del^2;
    ceq  = [];
    GC   = 2*x;
    GCeq = [];
end

