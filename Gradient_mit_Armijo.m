%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File    : Gradient_mit_Armijo.m                                         %
%                                                                         %
% Author  : Tobias Holicki                                                %
% Version : 01                                                            %
% Date    : 31.10.2018                                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Diese Funktion realisiert das Gradientenverfahren mit Armijo
% Schrittweitenregel.
%

function [z, fz] = Gradient_mit_Armijo(f, Grad_f, z0, beta, gamma)

% Initialisierungen und Abkürzungen zum Zeitsparen
z        = z0;
fz       = f(z0);
gradf    = Grad_f(z0);
normgrad = norm(gradf, 2)^2;

% Abbruch, falls der Gradient in der Norm klein genug ist 
% (||gradf|| <= 10^(-7))
while normgrad > 1e-14
    
    % Initialisierung
    sk = 1 / beta;
    
    % Schrittweite berechnen
    while 1
        % reduziere Schrittweite
        sk = beta * sk;
        
        % Berechne Updates
        z_neu  = z(:, end) - sk * gradf;
        fz_neu = f(z_neu); 

        % Abbruch, falls Armijo Bedingung nicht erfüllt ist
        if fz_neu <= fz(end) - gamma * sk * normgrad
           break; 
        end
    end

    % Ergebnisse sammeln und Abkürzungen updaten
    z        = [z, z_neu];       % Neue Iterte anhängen
    fz       = [fz, fz_neu];     % Neue Kosten anhängen
    gradf    = Grad_f(z_neu);    % Gradientenauswertung festhalten   
    normgrad = norm(gradf, 2)^2; % Norm vom Gradienten festhalten
end

end
