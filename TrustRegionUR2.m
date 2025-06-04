%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File    : TrustRegionUR.m                                               %
%                                                                         %
% Author  : Tobias Holicki                                                %
% Version : 01                                                            %
% Date    : 09.01.2020                                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Diese Funktion realisiert das Unterraum Trust-Region Verfahren
% für die Wahl V_k = span(Grad f(x_k), -Hess f(x_k)^{-1} Grad f(x_k)).
%

function [z, fz] = TrustRegionUR2(f, gradf, Hessf, z0, Del0, eta1, eta2, ...
                        gamma1, gamma2, Delmin)

% Initialisierungen und Abkürzungen zum Zeitsparen
z         = z0;
Del       = Del0;
fz        = f(z0);
gradfk    = gradf(z0);
normgradk = norm(gradfk, 2);
Hessfk    = Hessf(z0);
eps       = 1e-7;

% Abbruch, falls der Gradient in der Norm klein genug ist 
% (||gradf|| <= eps)
while normgradk > eps
    %% Suchrichtung bestimmen 
    
    % Neue Richtung bestimmen
    dk = NeueRichtung(gradfk, Hessfk, Del(end));

    % Potenzielle Updates
    qkdk   = fz(end) + gradfk' * dk + 0.5 * dk' * Hessfk * dk;
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


function dk = NeueRichtung(gradfk, Hessfk, Del)

options   = optimoptions('fmincon','Display', 'off', ...
                         'SpecifyObjectiveGradient', true, ...
                         'SpecifyConstraintGradient', true);
                     
% Unterraum bauen (wir setzen voraus, dass Hessfk invertierbar ist)
W = [gradfk,  -Hessfk \ gradfk];    


q = @(t) objective(t, W'*gradfk, W'*Hessfk*W);
c = @(t) constraints(W, t, Del);
[dk, ~] = fmincon(q, [0;0], [], [], [], [], [], [], c, options);

% Wieder fürs ursprüngliche Problem anpassen
dk = W * dk;
end

function [f, g] = objective(t, g, H)
     f = g' * t + 0.5 * t' * H * t;
     g = H * t + g;
end


function [c,ceq, GC, GCeq] = constraints(W, t, Del)
         c    = norm(W * t, 2)^2 - Del^2;
         ceq  = [];
         GC   = 2*W'*W * t;
         GCeq = [];
end

