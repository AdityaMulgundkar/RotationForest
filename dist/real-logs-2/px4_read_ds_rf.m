motor = 6;
dname = strcat('m', string(motor), '/');
fnames = ls(strcat('m',string(motor), '/'));
% fnames = fnames(1,:);
str = [];
itemLen = length(fnames(:,1));
for i=3:(itemLen)
str = [str convertCharsToStrings(fnames(i,:))];
end

for i=1:length(str)
    fname = str(i);
    ulog = ulogreader(strcat(dname, fname));
    msg = readTopicMsgs(ulog);
    
    % Fault time is the fault timestamp
    start_time = 4;
    sample_time = 4;
    
    dstart = ulog.StartTime;
    dend = ulog.EndTime;
    d1 = ulog.StartTime;
    d2 = ulog.EndTime;
    
    data2 = readTopicMsgs(ulog,'TopicNames',{'vehicle_angular_velocity',}, ... 
    'InstanceID',{0},'Time',[d1 d2]);
    vehicle_angular_velocity = data2.TopicMessages{1,1};
    
    data3 = readTopicMsgs(ulog,'TopicNames',{'vehicle_rates_setpoint',}, ... 
    'InstanceID',{0},'Time',[d1 d2]);
    vehicle_rates_setpoint = data3.TopicMessages{1,1};
    
    attR_d = rad2deg((vehicle_rates_setpoint.roll));
    attP_d = rad2deg((vehicle_rates_setpoint.pitch));
    attY_d = rad2deg((vehicle_rates_setpoint.yaw));
    
    angular_velocity = vehicle_angular_velocity;
    
    x = angular_velocity.xyz(:,2); % X -> Y
    y = -angular_velocity.xyz(:,1); % Y -> -X
    z = -angular_velocity.xyz(:,3);
    % Convert from FRD to NED frame
    angular_velocity_ned = [rad2deg(x) rad2deg(y) rad2deg(z)]; % Z -> -Z
    attR = angular_velocity_ned(:,1);
    attP = angular_velocity_ned(:,2);
    attY = angular_velocity_ned(:,3);

    
    timeHolder = attR;
    
    fault = double(attR);
    RATELastTimestamp = 340;
    
    for i=1:length(fault)
        if(i > RATELastTimestamp)
            fault(i) = 0.0;
        else
            fault(i) = 0.0;
        end
    end
    
    headers = ["R", 'RDes', 'P', 'PDes', 'Y', 'YDes', 'FaultIn'];
    
    writematrix(headers,strcat(dname, fname, '.csv'));
        writematrix([...
            (attR),...
            (attR_d),...
            (attP),...
            (attP_d),...
            (attY),...
            (attY_d),...
            (fault),...
            ],strcat(dname, fname, '.csv'),'WriteMode','append');
end