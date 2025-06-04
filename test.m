%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File    : test.m                                                        %
%                                                                         %
% Author  : Tobias Holicki                                                %
% Version : 01                                                            %
% Date    : 09.01.2020                                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Dieses Skript dient als Test für das Trust Region Verfahren.
% 

% Aufräumen
clc
clear
%close all

% Funktion mit symbolischen Variablen definieren
syms x y z;
f = sqrt((x-1)^2+10*(y-1)^2+100*(z-1)^2 + 1); 
% Rosenbrock Funktion
%f =  100 * (y - x^2)^2 +(1 - x)^2 + 100 * (z - y^2)^2 +(1 - y)^2;  

% Gradient und Hessematrix berechnen lassen
gradf(x,y,z) = gradient(f, [x, y, z]);
Hessf(x,y,z) = hessian(f, [x, y, z]);

% Konvertiere die symbolischen Funktionen in function handles zum Rechnen
tf     = matlabFunction(f);
tgradf = matlabFunction(gradf);
tHessf = matlabFunction(Hessf);

% Jetzt noch mit Vektoren als Eingabeparameter...
f     = @(x) tf(x(1), x(2), x(3));
gradf = @(x) tgradf(x(1), x(2), x(3));
Hessf = @(x) tHessf(x(1), x(2), x(3));

% Gegebene Parameter
beta  = 0.5;    % Armijo-Parameter
gamma = 0.01;   % Armijo-Parameter
alpha = 1e-6;   % Winkelbedingungsparameter
p     = 1;      % Winkelbedingungsparameter

% Trust Region Parameter
eta1   = 0.1;
eta2   = 0.75;
gamma1 = 0.5;
gamma2 = 2;
Delmin = 0.1;


%% Teil (b)

% Startwert
z0 = [0.5; 0.5; 0.5];

Del0 = 0.5;

[z, fz] = TrustRegionUR(f, gradf, Hessf, z0, Del0, eta1, eta2, ...
                         gamma1, gamma2, Delmin);

% Ergebnisse ausgeben
z_opt = z(:, end)
f_min = fz(end)


%% Teil (c)

% Optimum, was hier einfach ablesbar ist
zopt  = [1; 1; 1]; 

% Iterierte für beide Startwerte und beide Verfahren bestimmen
B0 = eye(3);

z0 = [0.95; 0.95; 0.95]; % Zeigt etwas schlechter als Newton
z0 = [0.5; 0.5; 0.5]; % Hier besser weil Newton zuerst nur Grad-Schritte

% Verfahren laufen lassen
[z, fz]   = Global_Newton(f, gradf, Hessf, z0, beta, gamma, alpha, p);
[za, fza] = Gradient_mit_Armijo(f, gradf, z0, beta, gamma);
[zb, fzb] = Globalisiertes_BFGS(f, gradf, z0, B0, gamma, 0.9);
[zc, fzc] = TrustRegionUR(f, gradf, Hessf, z0, Del0, eta1, eta2, ...
                          gamma1, gamma2, Delmin);
[zd, fzd] = TrustRegionVoll(f, gradf, Hessf, z0, Del0, eta1, eta2, ...
                          gamma1, gamma2, Delmin);

% Fehlerberechnung
err_z   = vecnorm(z - zopt, 2);
err_za  = vecnorm(za - zopt, 2);
err_zb  = vecnorm(zb - zopt, 2);
err_zc  = vecnorm(zc - zopt, 2);
err_zd  = vecnorm(zd - zopt, 2);


% Ergebnisse plotten
figure
semilogy(err_z)
hold on;
semilogy(err_za(1:length(err_zb)))
semilogy(err_zb)
semilogy(err_zc)
semilogy(err_zd)
legend('||z_n - z_{opt}|| via Newton', ...
       '||z_n - z_{opt}|| via Gradient', ...
       '||z_n - z_{opt}|| via BFGS', ...
       '||z_n - z_{opt}|| via Trust Region UR', ...
       '||z_n - z_{opt}|| via Trust Region Voll', ...
       'Location','Northeast');
title(['z0 = [ ', num2str(z0(1)), ', ', num2str(z0(2)), ...
               ', ', num2str(z0(3)), ']']);
