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

function [z, fz] = TrustRegionUR(f, gradf, Hessf, z0, Del0, eta1, eta2, ...
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
    
% Unterraum bauen (wir setzen voraus, dass Hessfk invertierbar ist)
w1 = gradfk;
w2 = -Hessfk \ gradfk;    

% Gram Schmitt
v1  = w1 / norm(w1, 2);
v2t = w2 - (v1' * w2) * v1;
v2  = v2t / norm(v2t);

% 2D Problem aufstellen
V = [v1, v2];

g = V' * gradfk;
H = V' * Hessfk * V;

% Vielleicht liegt das Minimum im inneren
dk = - H \ g;
e = min(eig(H)); % dk könnte auch Sattelpunkt sein

if norm(dk) > Del || e < 0 % Dann muss man Anpassungen vornehmen und es
                           % wird etwas unschön
   
    syms la;
    tH = H + 2 * la * eye(2);
    d = - [tH(2, 2), -tH(1, 2); -tH(1, 2), tH(1, 1)] * g;
    p = d(1)^2 + d(2)^2 - Del^2 * det(tH)^2;
    r = roots(fliplr(double(coeffs(p))));
    
    r = r(abs(imag(r)) < eps);

    for i = 1 : length(r)
        td{i} = -(H + 2*r(i) *eye(2)) \ g;
        t(i) = g' * td{i} + 0.5*td{i}' * H * td{i};
    end
    [~, j] = min(t);
    
    dk = td{j};
end

% Wieder fürs ursprüngliche Problem anpassen
dk = V * dk;

end

