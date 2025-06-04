%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File    : Global_Newton.m                                               %
%                                                                         %
% Author  : Tobias Holicki                                                %
% Version : 01                                                            %
% Date    : 05.12.2018                                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Diese Funktion realisiert das globalisierte Newton-Verfahren 
% (Algorithmus 2.71 aus dem Skript zur Einführung in die Optimierung)
%

function [z, fz] = Global_Newton(f, gradf, Hessf, z0, beta, gamma, alpha,p)

% Initialisierungen und Abkürzungen zum Zeitsparen
z         = z0;
fz        = f(z0);
gradfk    = gradf(z0);
normgradk = norm(gradfk, 2);
Hessfk    = Hessf(z0);

% Abbruch, falls der Gradient in der Norm klein genug ist 
% (||gradf|| <= 10^(-7))
while normgradk > 1e-7
    %% Suchrichtung bestimmen 
    % Prüfe, ob die aktuelle Hessematrix invertierbar ist
    if min(svd(Hessfk)) > 1e-5
        % Potentielle neue Abstiegsrichtung
        dk = - Hessfk \ gradfk; % Gleichungssystem lösen
        
        % Winkelbedingung prüfen
        if - gradfk' * dk < alpha * normgradk^(p+1) * norm(dk, 2)
            dk = -gradfk; % Richtung aus dem Gradientenverfahren
        else
            % Richtung aus dem Newton-Verfahren behalten
        end
    else
        dk = - gradfk; % Richtung aus dem Gradientenverfahren
    end
    
    %% Schrittweite bestimmen
    % Initialisierung
    sk = 1 / beta;
    
    % Schrittweite berechnen
    while 1
        % reduziere Schrittweite
        sk = beta * sk;
        
        % Berechne Updates
        z_neu  = z(:, end) + sk * dk;
        fz_neu = f(z_neu); 
       
        % Abbruch, falls Armijo Bedingung nicht erfüllt ist
        if fz_neu <= fz(end) + gamma * sk * gradfk' * dk
           break; 
        end
    end
    
    %% Ergebnisse sammeln und Abkürzungen updaten
    z         = [z, z_neu];      % Neue Iterte anhängen
    fz        = [fz, fz_neu];    % Neue Kosten anhängen
    gradfk    = gradf(z_neu);    % Gradientenauswertung festhalten   
    normgradk = norm(gradfk, 2); % Norm vom Gradienten festhalten
    Hessfk    = Hessf(z_neu);    % Hessematrixauswertung festhalten
end

end
