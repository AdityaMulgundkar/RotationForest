clear;

% Choose which log file to read
filename = 'm3_log.mat';

% Choose variables to read - you can open the .mat file and read the (upto
% 4 digit) var names. You can add the desired var names here.
varsToRead = {'ATT','RATE','PARM'};

DATASET = load(filename, varsToRead{:});

% Change searchParam value; by replacing the 1 to any other integer in
% [1,2,3,4] for QuadCopter or [1,2,3,4,5,6] for HexaCopter
searchParam = 'SERVO1_FUNCTION';

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
key = 208;
RATELastTimestamp=selectedArray.TimeUS(key);
timeHolder = selectedArray.TimeUS;

rHolder = selectedArray.R;
rHolder2 = awgn(rHolder, 50);

fault = double(timeHolder);

for i=1:length(fault)
    if(fault(i) > RATELastTimestamp)
        fault(i) = 1.0;
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
plot(x, rateP, 'LineWidth', 2);
hold on;
plot(x, ratePDes, 'Color',[1,0.7,0], 'LineStyle', '--', 'LineWidth', 1);
line([x(208) x(208)], ylim, 'Color',[1,0,0], 'LineWidth', 0.5);
xlabel(t1,'Time in seconds');
ylabel(t1,'Pitch and Pitch Desired');
legend('Pitch','PitchDes','ERR','Location','northwest');
set(gca,'FontSize', 14);

x2 = transpose((selectedArray.R));

% Segregate roll, desroll, p and y too
% csvwrite('test1.csv', (selectedArray));
headers = ["R", 'RDes', 'P', 'PDes', 'Y', 'YDes', 'FaultIn'];

writematrix(headers,strcat('dist/m3/real-test.csv'));
    writematrix([...
        transpose(selectedArray.R),...
        transpose(selectedArray.RDes),...
        transpose(selectedArray.P),...
        transpose(selectedArray.PDes),...
        transpose(selectedArray.Y),...
        transpose(selectedArray.YDes),...
        transpose(fault)...
        ],strcat('dist/m3/real-test.csv'),'WriteMode','append');