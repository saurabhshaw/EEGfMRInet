% Set up UDP connection
udpReceiver = udp('127.0.0.1', 7000, 'LocalPort', 5006);
set(udpReceiver, 'InputBufferSize', 1024); % Adjust buffer size as needed
fopen(udpReceiver);

% Create figure and bar graph
figure;
hBar = bar(nan);
xlabel('Data Index');
ylabel('Data Value');
title('Real-time Bar Graph from UDP');

% Set up timer for real-time plotting
updateRate = 0.1; % Update every 0.1 seconds
timerObj = timer('TimerFcn', @(~,~) updateBarGraph(udpReceiver, hBar), 'Period', updateRate, 'ExecutionMode', 'fixedRate');
start(timerObj);

% Wait for user to close the figure
waitfor(gcf);

% Clean up
fclose(udpReceiver);
delete(timerObj);

function updateBarGraph(udpReceiver, hBar)
    % Read data from UDP port
    dataReceived = fread(udpReceiver, udpReceiver.BytesAvailable, 'double');
    
    % Update bar graph
    if ~isempty(dataReceived)
        set(hBar, 'YData', dataReceived);
        drawnow;
    end
end