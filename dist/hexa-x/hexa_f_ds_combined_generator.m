clear;
motorNum = 5;
% Choose which log file to read
filename = strcat('m',num2str(motorNum),'.mat');

% Choose variables to read - you can open the .mat file and read the (upto
% 4 digit) var names. You can add the desired var names here.
varsToRead = {'ATT','RATE','PARM'};

DATASET = load(filename, varsToRead{:});

% Change searchParam value; by replacing the 1 to any other integer in
% [1,2,3,4] for QuadCopter or [1,2,3,4,5,6] for HexaCopter
searchParam = strcat('SERVO',num2str(motorNum),'_FUNCTION');

searchParamUses = ~cellfun('isempty',strfind(cellstr(DATASET.PARM.Name),searchParam));
searchLastParamUse = find(searchParamUses);

% Index of Last usage of param stored here
lastParamUse = searchLastParamUse(end);

% Timestamp (TimeUS) variable of the above packet - this is the time in
% microseconds, when the motor is overridden. Fault occurs immediately
% after this.
lastParamTimestamp = DATASET.PARM.TimeUS(lastParamUse);

% Select the dataset array you wish to work on (Baro, Gyro, EKF, ...)
selectedArray = DATASET.RATE;

[val, key] = min(abs(selectedArray.TimeUS-lastParamTimestamp));
RATELastTimestamp=selectedArray.TimeUS(key);
timeHolder = selectedArray.TimeUS;

cutoff = key + 10;

rHolder = selectedArray.R;
rHolder2 = awgn(rHolder, 50);

fault = double(timeHolder);

for i=1:length(fault)
    if(fault(i) > RATELastTimestamp)
        fault(i) = motorNum;
    else
        fault(i) = 0.0;
    end
end

% TODO: Marker
disp("RATELastTimestamp: " + RATELastTimestamp);
timestampInSeconds = RATELastTimestamp*10^-6;

x = (selectedArray.TimeUS);
f= figure;
rateP = selectedArray.P;
ratePDes = selectedArray.PDes;
t1 = nexttile;
plot(x*10^-6, rateP, 'LineWidth', 2);
hold on;
plot(x*10^-6, ratePDes, 'Color',[1,0.7,0], 'LineStyle', '--', 'LineWidth', 1);
% line([782 782], ylim, 'Color',[1,0,0], 'LineWidth', 0.5);
line([timestampInSeconds timestampInSeconds], ylim, 'Color',[1,0,0], 'LineWidth', 0.5);
xlabel(t1,'Time in seconds');
ylabel(t1,'Pitch and Pitch Desired');
legend('Pitch','PitchDes','ERR','Location','northwest');
set(gca,'FontSize', 14);

x2 = transpose((selectedArray.R));

% Segregate roll, desroll, p and y too
% csvwrite('test1.csv', (selectedArray));
headers = ["R", 'RDes', 'P', 'PDes', 'Y', 'YDes', 'FaultIn'];

sigR = 90;

x1 = selectedArray.R(1:cutoff);

for i=1:10
    writematrix(headers,strcat('err10/m',num2str(motorNum),'/test',num2str(i),'.csv'));
    writematrix([...
        awgn(transpose(selectedArray.R(1:cutoff)), sigR),...
        awgn(transpose(selectedArray.RDes(1:cutoff)), sigR),...
        awgn(transpose(selectedArray.P(1:cutoff)), sigR),...
        awgn(transpose(selectedArray.PDes(1:cutoff)), sigR),...
        awgn(transpose(selectedArray.Y(1:cutoff)), sigR),...
        awgn(transpose(selectedArray.YDes(1:cutoff)), sigR),...
        transpose(fault(1:cutoff))...
        ],strcat('err10/m',num2str(motorNum),'/test',num2str(i),'.csv'),'WriteMode','append');
end