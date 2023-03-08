% M1 fault @ 365
ulog = ulogreader('log_457_2023-3-6-09-43-44.ulg');

% M3 fault @ 552
%ulog = ulogreader('log_481_2023-3-6-11-30-18.ulg');

msg = readTopicMsgs(ulog);
motor_num = 3;

% Fault time is the fault timestamp
faulTimestamp = 365;
%faulTimestamp = 552;

dstart = ulog.StartTime;
dend = ulog.EndTime;

data2 = readTopicMsgs(ulog,'TopicNames',{'vehicle_angular_velocity',}, ... 
'InstanceID',{0},'Time',[dstart dend]);

vehicle_angular_velocity = data2.TopicMessages{1,1};

data3 = readTopicMsgs(ulog,'TopicNames',{'vehicle_rates_setpoint',}, ... 
'InstanceID',{0},'Time',[dstart dend]);

vehicle_rates_setpoint = data3.TopicMessages{1,1};

attR = rad2deg(vehicle_angular_velocity.xyz(:,1));
attP = rad2deg(vehicle_angular_velocity.xyz(:,2));
attY = rad2deg(vehicle_angular_velocity.xyz(:,3));
attR_d = rad2deg((vehicle_rates_setpoint.roll));
attP_d = rad2deg((vehicle_rates_setpoint.pitch));
attY_d = rad2deg((vehicle_rates_setpoint.yaw));

timeHolder = attR;

fault = double(attR);

for i=1:length(fault)
    if(i > faulTimestamp)
        fault(i) = 1.0;
    else
        fault(i) = 0.0;
    end
end

headers = ["R", 'RDes', 'P', 'PDes', 'Y', 'YDes', 'FaultIn'];

writematrix(headers,strcat('m',num2str(motor_num),'/real-test.csv'));
    writematrix([...
        (attR),...
        (attR_d),...
        (attP),...
        (attP_d),...
        (attY),...
        (attY_d),...
        (fault)...
        ],strcat('m',num2str(motor_num),'/real-test.csv'),'WriteMode','append');