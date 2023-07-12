T = readtable('graphs/V-L1/m1.csv');

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
plot(x, circshift(T.R,1), 'Color',[0,0,1], 'LineStyle', '--');
hold on;
plot(x, T.RDes, 'Color',[0,0,1], 'LineStyle', '-.');
hold on;
ylim([-120 35]);
line([172 172], ylim, 'Color',[1,0,0], 'LineStyle', '-.', 'LineWidth', 1);
xlim([171 180]);
xlabel('Sample Number');
%ylabel({'$R$,$R-1$,$R_d$ (in deg)';'Classifier C2-A'}, 'Interpreter','LaTeX');
ylabel({'$\omega_x(k), \omega_x(k-1), \omega_{xd}(k)$ (in deg/s)';'Classifier C2-A'}, 'Interpreter','LaTeX');
%legend('$R$','$R-1$','$R_d$','Fault introduced','Fault classified','Location','northeast', 'Interpreter','LaTeX');
legend('$\omega_x(k)$','$\omega_x(k-1)$','$\omega_{xd}(k)$','Fault classified','Location','northeastoutside', 'Interpreter','LaTeX');
set(gca,'FontSize', 12);
ax = gca().XAxis;
ay = gca().YAxis;
ax.Color = color2;
ay.Color = color2;
ax.Label.Color = [0, 0, 0];
ay.Label.Color = [0, 0, 0];
ax.TickLabelColor = [0,0,0];
ay.TickLabelColor = [0,0,0];