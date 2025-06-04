%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File    : Globalisiertes_BFGS.m                                         %
%                                                                         %
% Author  : Tobias Holicki                                                %
% Version : 01                                                            %
% Date    : 08.01.2019                                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Diese Funktion realisiert das globalisierte BFGS Verfahren.
%
function [z, fz] = Globalisiertes_BFGS(f, gradf, z0, B0, gamma, eta)

% Initialisierungen und Abkürzungen zum Zeitsparen
z        = z0;
fz       = f(z0);
gradfk   = gradf(z0);
normgrad = norm(gradfk, 2);
Bk       = B0;
eps      = 1e-7;
% Abbruch, falls der Gradient in der Norm klein genug ist 
% (||gradf|| <= 10^(-7))
while normgrad > eps
    %normgrad
    
    %% Abstiegsrichtung
    dk = -Bk * gradfk;
    
    %% Schrittweite berechnen mit Powell-Wolfe-Regel
    sm = 2;
    while 1
       sm = sm/2;
       % Berechne Updates
       z_neu  = z(:, end) + sm * dk;
       fz_neu = f(z_neu);
       % Abbruch, falls Armijo Bedingung erfüllt ist
       if fz_neu <= fz(end) + gamma * sm * gradfk' * dk
           break;
       end
    end
    
    sp = sm;
    while 1
        sp = sp * 2;
        % Berechne Updates
        z_neu  = z(:, end) + sp * dk;
        fz_neu = f(z_neu); 
        % Abbruch, falls Armijo Bedingung nicht erfüllt is
        if fz_neu > fz(end) + gamma * sp * gradfk' * dk
            break;
        end
    end
    
    % Nochmal Updates
    z_neu  = z(:, end) + sm * dk;
    gradf_neu = gradf(z_neu);
    while gradf_neu' * dk < eta * gradfk' * dk
        s0 = (sp + sm) / 2;
        % Und wieder updates
        z_neu  = z(:, end) + s0 * dk;
        fz_neu = f(z_neu); 
        if fz_neu <= fz(end) + gamma * s0 * gradfk' * dk
            sm = s0;
            gradf_neu = gradf(z_neu);
        else
            sp = s0;
        end
    end
    
    % Finale Schrittweite
    sk = sm;
    
    %% Updates
    z_neu  = z(:, end) + sk * dk;
    fz_neu = f(z_neu); 
    gradf_neu = gradf(z_neu);
    normgrad = norm(gradf_neu, 2);
    
    if normgrad > eps
        tdk = sk * dk;
        yk  = gradf_neu - gradfk;
        te  = (tdk - Bk * yk) * tdk';
        te2 = (tdk - Bk * yk)' * yk;
        te3 = tdk' * yk;
        Bk = Bk + (te + te') / te3 - te2 * (tdk * tdk') / te3^2;
    else
        z = [z, z_neu];       % Neue Iterte anhängen
        fz = [fz, fz_neu];     % Neue Kosten anhängen
        break;
    end
    
    % Ergebnisse sammeln und Abkürzungen updaten
    z        = [z, z_neu];       % Neue Iterte anhängen
    fz       = [fz, fz_neu];     % Neue Kosten anhängen
    gradfk   = gradf(z_neu);    % Gradientenauswertung festhalten   
end

end
