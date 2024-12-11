% Parámetros iniciales
F0 = 1;              % Amplitud máxima
A = 0.5;             % Modulación del período
Delta_t = 0.02;      % Anchura de las gaussianas
N = 10;              % Número de términos en la suma
t = linspace(0, 5, 1000); % Intervalo de tiempo

% Definición de los picos t_n
t_n = 2 * F0 * A * (0:N); % Generar las posiciones t_n

% Inicialización de las funciones
f_gaussianas = zeros(size(t)); % Para la suma de gaussianas

% Cálculo de f(t) basado en la suma de gaussianas
for n = 1:(N + 1)
    f_gaussianas = f_gaussianas + F0 * exp(-0.5 * ((t - t_n(n)) / Delta_t).^2);
end

% Cálculo de f(t) basado en coseno cuadrado (frecuencia sincronizada)
omega = pi / (2 * F0 * A); % Frecuencia angular
f_cos2 = F0 * cos(omega * t).^2;

% Graficar
figure;
plot(t, f_gaussianas, 'b', 'LineWidth', 1.5); % Gráfica de gaussianas
hold on;
plot(t, f_cos2, 'r--', 'LineWidth', 1.5);     % Gráfica de coseno cuadrado

% Etiquetas y configuración
xlabel('Time $t$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('$F(t)$', 'Interpreter', 'latex', 'FontSize', 12);
title('Comparison of $f(t)$: Gaussians vs Squared Cosine', 'Interpreter', 'latex', 'FontSize', 14);
grid on;
grid minor;
axis([0 4 0 1.2]); % Ajuste de ejes

legend({'Sum of Gaussians', 'Squared Cosine'}, 'Interpreter', 'latex', 'FontSize', 12);
hold off;


