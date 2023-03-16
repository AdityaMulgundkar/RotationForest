%% IEEE Standard Figure Configuration - Version 1.0

% run this code before the plot command

%%
% According to the standard of IEEE Transactions and Journals: 

% Times New Roman is the suggested font in labels. 

% For a singlepart figure, labels should be in 8 to 10 points,
% whereas for a multipart figure, labels should be in 8 points.

% Width: column width: 8.8 cm; page width: 18.1 cm.

%% width & hight of the figure
k_scaling = 5;          % scaling factor of the figure
% (You need to plot a figure which has a width of (8.8 * k_scaling)
% in MATLAB, so that when you paste it into your paper, the width will be
% scalled down to 8.8 cm  which can guarantee a preferred clearness.
k_width_hight = 2;      % width:hight ratio of the figure

width = 8.8 * k_scaling;
hight = width / k_width_hight;

%% figure margins
top = 0.5;  % normalized top margin
bottom = 3;	% normalized bottom margin
left = 3.5;	% normalized left margin
right = 1;  % normalized right margin

%% set default figure configurations
set(0,'defaultFigureUnits','centimeters');
set(0,'defaultFigurePosition',[0 0 width hight]);

set(0,'defaultLineLineWidth',1*k_scaling);
set(0,'defaultAxesLineWidth',0.25*k_scaling);

set(0,'defaultAxesGridLineStyle',':');
set(0,'defaultAxesYGrid','on');
set(0,'defaultAxesXGrid','on');

set(0,'defaultAxesFontName','Times New Roman');
set(0,'defaultAxesFontSize',9*k_scaling);

set(0,'defaultTextFontName','Times New Roman');
set(0,'defaultTextFontSize',9*k_scaling);

set(0,'defaultLegendFontName','Times New Roman');
set(0,'defaultLegendFontSize',10*k_scaling);

set(0,'defaultAxesUnits','normalized');
set(0,'defaultAxesPosition',[left/width bottom/hight (width-left-right)/width  (hight-bottom-top)/hight]);

set(0,'defaultAxesColorOrder',[0 0 0]);
set(0,'defaultAxesTickDir','out');

set(0,'defaultFigurePaperPositionMode','auto');

% you can change the Legend Location to whatever as you wish
set(0,'defaultLegendLocation','southeast');
set(0,'defaultLegendBox','on');
set(0,'defaultLegendOrientation','vertical');

% Define the data
data = [
    0.97 0.86 0.96 0.96 0.96;
    0.92 0.80 0.94 0.89 0.88;
    0.89 0.83 0.88 0.85 0.89;
    0.84 0.79 0.84 0.83 0.87;
    0.81 0.81 0.79 0.79 0.83;
    0.82 0.78 0.81 0.82 0.84;
];

% Define the names for each category
categories = {'Logistic Regression', 'Gaussian Naive Bayes', 'AdaBoost', 'Random Forest', 'Rotation Forest'};

% Define the names for each data point
names = {'Motor 1', 'Motor 2', 'Motor 3', 'Motor 4', 'Motor 5', 'Motor 6'};

% Plot the radar chart
figure;
polarplot(0:2*pi/5:2*pi, [data(1,:) data(1,1)], '-o', 'LineWidth', 2.3, 'MarkerSize', 8);
hold on;
for i = 2:size(data,1)
    polarplot(0:2*pi/5:2*pi, [data(i,:) data(i,1)], '-o', 'LineWidth', 2.3, 'MarkerSize', 8);
end
hold off;
rlim([0.75,1]);
set(gca, 'ThetaTick', linspace(0, 360, length(categories) + 1), 'ThetaTickLabel', [categories categories{1}]); % Set the angular tick marks
%title('Comparison of Classification Models for Motor Control', 'FontSize', 20); % Set the title font size
legend(names, 'Location', 'bestoutside', 'FontSize', 18); % Set the legend font size
ax = gca;
ax.FontSize = 18; % Set the axis label font size


%{

% ulog1 = ulogreader('Hexa-X-Motor1.ulg');
opts = detectImportOptions('csv_file.csv');
% preview('Tdata.csv',opts);
M = readmatrix('csv_file.csv');
headers = opts.SelectedVariableNames;

% Fault time is the fault timestamp
% m1
fault_time1 = duration("00:01:14.033282",'Format','hh:mm:ss.SSSSSS');

d1 = duration(fault_time1,'Format','hh:mm:ss.SSSSSS') - duration([0 0 7],'Format','hh:mm:ss.SSSSSS');
df = duration(fault_time1,'Format','hh:mm:ss.SSSSSS');
d2 = duration(fault_time1,'Format','hh:mm:ss.SSSSSS') + duration([0 0 7],'Format','hh:mm:ss.SSSSSS');

x1 = zeros(size(M(:,1)));

for c = 1:length(x1)
    x1(c) = c;
end

p = M(:, find(headers=="R"));
pd = M(:, find(headers=="Rdes"));
q = M(:, find(headers=="P"));
qd = M(:, find(headers=="Pdes"));
r = M(:, find(headers=="Y"));
rd = M(:, find(headers=="Ydes"));

f1 = M(:, find(headers=="f1"));
f2 = M(:, find(headers=="f2"));
f3 = M(:, find(headers=="f3"));

G = [f1,f2,f3]


l1 = 217;
l2 = 220;
l3 = 219;
l4 = 221;

xmin = 100;
xmax = 300;
bar(G, 'grouped')

xlabel('Test Number');
ylabel('\Delta T_f (1 unit = 5ms)');
%set(gca,'FontSize', 20);
legend("f1", "f2", "f3");



figure;

plot(x1, p+1500, 'b');
hold on;
plot(x1, pd +1500, 'g');
hold on;
plot(x1, q+700, 'r');
hold on;
plot(x1, r, 'm');
hold on;
line([l1 l1], ylim,  'LineWidth', 1.5, 'LineStyle', '--');
line([l2 l2], ylim, 'Color',[1,0,0], 'LineWidth', 1.5, 'LineStyle', '--');
xlim([xmin, xmax]);
hold off;
xlabel('Sample number (1 unit = 5ms)');
ylabel('angle (-180^o, 180^o)');
%set(gca,'FontSize', 20);
legend('\omega_x','\omega_{xd}','\omega_y','\omega_z', 'T_{f1}', 'T_{f2}');
yticks([]);


figure;
plot(x1, p+1500, 'b');
hold on;
plot(x1, q+700, 'r');
hold on;
plot(x1, qd+700, 'g');
hold on;
plot(x1, r, 'm');
hold on;
line([l1 l1], ylim,  'LineWidth', 1.5, 'LineStyle', '--');
line([l3 l3], ylim, 'Color',[1,0,0], 'LineWidth', 1.5, 'LineStyle', '--');
xlim([xmin, xmax]);
hold off;
xlabel('Sample number (1 unit = 5ms)');
ylabel('angle (-180, 180)');
%set(gca,'FontSize', 20);
legend('\omega_x','\omega_y','\omega_{yd}','\omega_z', 'T_{f1}', 'T_{f2}');
yticks([]);


figure;
plot(x1, p+1500, 'b');
hold on;
plot(x1, q+700, 'r');
hold on;
plot(x1, r, 'm');
hold on;
plot(x1, rd, 'g');
hold on;
line([l1 l1], ylim,  'LineWidth', 1.5, 'LineStyle', '--');
line([l4 l4], ylim, 'Color',[1,0,0], 'LineWidth', 1.5, 'LineStyle', '--');
xlim([xmin, xmax]);
hold off;
xlabel('Sample number (1 unit = 5ms)');
ylabel('angle (-180, 180)');
%set(gca,'FontSize', 20);
legend('\omega_x','\omega_y','\omega_z','\omega_{zd}','T_{f1}', 'T_{f2}');
yticks([]);

%gggggggggggggggggggggggggggggggggg
subplot(3,2,1);
hold on;
plot(x1, p, 'LineWidth', 1.25);
plot(x1, pd, 'LineWidth', 1.25);
line([l1 l1], ylim, 'Color',[1,0,0], 'LineWidth', 1, 'LineStyle', '--');
line([l2 l2], ylim, 'Color',[0,1,0], 'LineWidth', 1, 'LineStyle', '--');
hold off;
xlabel('Sample number');
ylabel('p, p_d');
set(gca,'FontSize', 14);
legend('p','p_d', 'Time of fault', 'Time of detection');

subplot(3,2,3);
hold on;
plot(x1, q, 'LineWidth', 1.25);
plot(x1, qd, 'LineWidth', 1.25);
line([l1 l1], ylim, 'Color',[1,0,0], 'LineWidth', 1, 'LineStyle', '--');
line([l2 l2], ylim, 'Color',[0,1,0], 'LineWidth', 1, 'LineStyle', '--');
hold off;
xlabel('Sample number');
ylabel('q, q_d');
set(gca,'FontSize', 14);
legend('q','q_d', 'Time of fault', 'Time of detection');

subplot(3,2,5);
hold on;
plot(x1, r, 'LineWidth', 1.25);
plot(x1, rd, 'LineWidth', 1.25);
line([l1 l1], ylim, 'Color',[1,0,0], 'LineWidth', 1, 'LineStyle', '--');
line([l2 l2], ylim, 'Color',[0,1,0], 'LineWidth', 1, 'LineStyle', '--');
hold off;
xlabel('Sample number');
ylabel('r, r_d');
set(gca,'FontSize', 14);
legend('r','r_d', 'Time of fault', 'Time of detection');


% trim
x1 = x1(200:240);
p = p(200:240);
pd = pd(200:240);
q = q(200:240);
qd = qd(200:240);
r = r(200:240);
rd = rd(200:240);


subplot(3,2,2);
hold on;
plot(x1, p, 'LineWidth', 1.25);
plot(x1, pd, 'LineWidth', 1.25);
line([l1 l1], ylim, 'Color',[1,0,0], 'LineWidth', 1, 'LineStyle', '--');
line([l2 l2], ylim, 'Color',[0,1,0], 'LineWidth', 1, 'LineStyle', '--');
hold off;
xlabel('Sample number');
ylabel('p, p_d');
set(gca,'FontSize', 14);
legend('p','p_d', 'Time of fault', 'Time of detection');

subplot(3,2,4);
hold on;
plot(x1, q, 'LineWidth', 1.25);
plot(x1, qd, 'LineWidth', 1.25);
line([l1 l1], ylim, 'Color',[1,0,0], 'LineWidth', 1, 'LineStyle', '--');
line([l2 l2], ylim, 'Color',[0,1,0], 'LineWidth', 1, 'LineStyle', '--');
hold off;
xlabel('Sample number');
ylabel('q, q_d');
set(gca,'FontSize', 14);
legend('q','q_d', 'Time of fault', 'Time of detection');

subplot(3,2,6);
hold on;
plot(x1, r, 'LineWidth', 1.25);
plot(x1, rd, 'LineWidth', 1.25);
line([l1 l1], ylim, 'Color',[1,0,0], 'LineWidth', 1, 'LineStyle', '--');
line([l2 l2], ylim, 'Color',[0,1,0], 'LineWidth', 1, 'LineStyle', '--');
hold off;
xlabel('Sample number');
ylabel('r, r_d');
set(gca,'FontSize', 14);
legend('r','r_d', 'Time of fault', 'Time of detection');


%}