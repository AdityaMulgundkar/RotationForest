T = readtable('graphs/V-L1/m3.csv');

x = T.Var1;
f = figure;

a = T.P;
b = circshift(T.P,-1);
t1 = nexttile;

str1 = '#95CCF7';
str2 = '#C6E7A6';
color1 = sscanf(str1(2:end),'%2x%2x%2x',[1 3])/255;
color2 = sscanf(str2(2:end),'%2x%2x%2x',[1 3])/255;

plot(x, T.R, 'Color',[0,0,1]);
hold on;
plot(x, T.P, 'Color',[1,0.5,0]);
hold on;
plot(x, circshift(T.P,1), 'Color',[1,0.5,0], 'LineStyle', '--');
hold on;
plot(x, T.PDes, 'Color',[1,0.5,0], 'LineStyle', '-.');
hold on;
plot(x, T.Y, 'Color',[0,0.8,0]);
hold on;
ylim([-180 110]);
line([172 172], ylim, 'Color',[1,0,0], 'LineStyle', '-.', 'LineWidth', 1);
xlim([171 180]);
xlabel('Sample Number');
ylabel({'$\omega_x(k)$,$\omega_y(k)$,$\omega_y(k-1)$,$\omega_{yd}(k)$,$\omega_z(k)$ (in deg/s)';'Classifier C2-B'}, 'Interpreter','LaTeX');
legend('$\omega_x(k)$','$\omega_y(k)$','$\omega_y(k-1)$','$\omega_{yd}(k)$','$\omega_z(k)$','Fault classified','Location','northeastoutside', 'Interpreter','LaTeX');
set(gca,'FontSize', 12);
ax = gca().XAxis;
ay = gca().YAxis;
ax.Color = color2;
ay.Color = color2;
ax.Label.Color = [0, 0, 0];
ay.Label.Color = [0, 0, 0];
ax.TickLabelColor = [0,0,0];
ay.TickLabelColor = [0,0,0];
