%% Cross validation script 
% This is for the cross validation study. Split dataset into 8 groups,
% train on 7 and test each of the rest. 
% ie 1st test: train on 2..7 and test on group 1
% 2nd test: train on 1,3,...7 and test on group 2
% and so on till trained on 1...7 and tested on 8th group. 
% Store accuracy results each time. 
% Final step is to average results.

%% Remove unecessary variables - keep tool signals, targets, and surgery times
clearvars -except toolInputs phaseTargets T_sur T_ph phaseTimeStamps

%% Take variables and convert data to useful format. 
[u_tools, y_phases, u, y] = narxdatafcn(toolInputs,phaseTargets,T_sur);


%% Data allocation

lsl = length(u_tools); % longest surgery length = 5993s

% Allocate tools to groups
u1 = cell(1,lsl); u2 = cell(1,lsl); u3 = cell(1,lsl); u4 = cell(1,lsl); 
u5 = cell(1,lsl); u6 = cell(1,lsl); u7 = cell(1,lsl); u8 = cell(1,lsl);

% Allocate targets to groups
y1 = cell(1,lsl); y2 = cell(1,lsl); y3 = cell(1,lsl); y4 = cell(1,lsl); 
y5 = cell(1,lsl); y6 = cell(1,lsl); y7 = cell(1,lsl); y8 = cell(1,lsl);

for k = 1: lsl 
    u1{k} = u_tools{1,k}(:,1:10);  y1{k} = y_phases{1,k}(1:10);
    u2{k} = u_tools{1,k}(:,11:20); y2{k} = y_phases{1,k}(11:20);
    u3{k} = u_tools{1,k}(:,21:30); y3{k} = y_phases{1,k}(21:30);
    u4{k} = u_tools{1,k}(:,31:40); y4{k} = y_phases{1,k}(31:40);
    u5{k} = u_tools{1,k}(:,41:50); y5{k} = y_phases{1,k}(41:50);
    u6{k} = u_tools{1,k}(:,51:60); y6{k} = y_phases{1,k}(51:60);
    u7{k} = u_tools{1,k}(:,61:70); y7{k} = y_phases{1,k}(61:70);
    u8{k} = u_tools{1,k}(:,71:80); y8{k} = y_phases{1,k}(71:80);
end

%% Groups 
% 1st cross-val study:
%   - train on groups 2:8
%   - test on group 1
cvTools1 = catsamples(u2,u3,u4,u5,u6,u7,u8,'pad');
cvPhases1 = catsamples(y2,y3,y4,y5,y6,y7,y8,'pad');

% 2nd cross-val study:
%   - train on groups 1,3..8
%   - test on group 2
cvTools2 = catsamples(u1,u3,u4,u5,u6,u7,u8,'pad');
cvPhases2 = catsamples(y1,y3,y4,y5,y6,y7,y8,'pad');
% 
% % 3rd cross-val study:
% %   - train on groups 1,2,4...8
% %   - test on group 3
cvTools3 = catsamples(u1,u2,u4,u5,u6,u7,u8,'pad');
cvPhases3 = catsamples(y1,y2,y4,y5,y6,y7,y8,'pad');
% 
% % 4th cross-val study:
% %   - train on groups 1,2,3,5...8
% %   - test on group 4
cvTools4 = catsamples(u1,u2,u3,u5,u6,u7,u8,'pad');
cvPhases4 = catsamples(y1,y2,y3,y5,y6,y7,y8,'pad');

% 5th cross-val study:
%   - train on groups 1...4,6...8
%   - test on group 5
cvTools5 = catsamples(u1,u2,u3,u4,u6,u7,u8,'pad');
cvPhases5 = catsamples(y1,y2,y3,y4,y6,y7,y8,'pad');

% 6th cross-val study:
%   - train on groups 1...5,7,8
%   - test on group 6
cvTools6 = catsamples(u1,u2,u3,u4,u5,u7,u8,'pad');
cvPhases6 = catsamples(y1,y2,y3,y4,y5,y7,y8,'pad');
% 
% 6th cross-val study:
%   - train on groups 1...6,8
%   - test on group 7
cvTools7 = catsamples(u1,u2,u3,u4,u5,u6,u8,'pad');
cvPhases7 = catsamples(y1,y2,y3,y4,y5,y6,y8,'pad');

% 6th cross-val study:
%   - train on groups 1...7
%   - test on group 8
cvTools8 = catsamples(u1,u2,u3,u4,u5,u6,u7,'pad');
cvPhases8 = catsamples(y1,y2,y3,y4,y5,y6,y7,'pad');

%% TRAIN THE NARX
% Create architecture
id = 3; 
fd = 3; 
hiddenNeurons = 12; 
narx_net = narxnet(1:id,1:fd,hiddenNeurons); 
narx_net.trainFcn = 'trainbr';
narx_net.performFcn = 'sse';
narx_net.layers{1}.transferFcn = 'tansig';
narx_net.layers{2}.transferFcn = 'purelin';
narx_net.divideFcn = 'dividetrain';  
narx_net.trainParam.epochs = 1000;
narx_net.trainParam.min_grad = 1e-6;
narx_net.trainParam.mu_max = 1e20;


% For this part, just change the dataset required for training.
[p,Pi,Ai,t] = preparets(narx_net,cvTools8,{},cvPhases8);
tic
[narx_net,tr] = train(narx_net,p,t,Pi);
tElapsed = toc/60;
trainTimeMins = floor(tElapsed) + (tElapsed-floor(tElapsed))*60/100; 

% Outputs
y_train = narx_net(p,Pi,Ai);

%% Predictions
% Use trained network for making predictions.
% This is the part where targets are removed, the open-loop is closed,
% and the network is allowed to prove its predictive ability on new data. 

% For first cv trial, only between 1 and 10, and so on for other trials
surg = input('Enter the surgery to be predicted : \n');
narx_netc = closeloop(narx_net); % create parallel architecture
[p1,Pi1,Ai1,t1] = preparets(narx_netc,u{surg},{},y{surg});
yp1 = narx_netc(p1,Pi1,Ai1);
TS = size(t1,2);

% Error between targets and predictions
e = zeros(1,TS);
for i = 1:TS
    e(i) = cell2mat(t1(i))-cell2mat(yp1(i));
end

% Plot targets against raw outputs. 
figure(1);
% subplot(2,1,1)
plot(1:TS,cell2mat(t1),'b',1:TS,cell2mat(yp1),'r');
title(['NARX network: surgery ' num2str(surg) ' phase predictions']);
xlabel('Time [s]');
ylabel('Phase Number');
legend('Actual','Predicted','Location','best');
% subplot(2,1,2);
% plot(e);

% -> plot is messy so need to tidy up with a heaviside function.

%% Filtered predictions
% Apply heaviside function to level off plot
% For easier visualisation of predictions, implement heaviside method to
% level off spikes.
predLength = length(yp1); % get length of matrix of predicted values
yp1f = predfilt(yp1,predLength); % filter values

% Plot cleaned results
figure(2);
plot(1:TS,cell2mat(t1),'b',1:TS,yp1f,'r');
title(['NARX network: surgery ' num2str(surg) ' phase predictions']);
xlabel('Time [s]');
ylabel('Phase Number');
legend('Actual','Predicted','Location','best');

%% Compute accuracy
% Store each in Numbers so that I can take average later 
% This section involves taking the fitered outputs, yp1f, and computing the
% confusion matrix associated with the corresponding targets of each
% surgery. Of course, different surgeries will have different accuracies
% and therefore the overall accuracy (for the 10 vids) will be the mean of
% all the accuracies.

load data.txt
T_ph = data(:,1:7);

% Shift the first phase by nTdl timesteps, NOT all the phases. 
nTdl = fd; % number of tapped delay lines
phaseTimes = T_ph(surg,:);
phaseTimes(1) = phaseTimes(1)-nTdl; 
t2 = cell2mat(t1); % convert cl targets from cell to matrix 
[percentClass,phaseCount,phaseDiff] = narxClassStats(t2,yp1f,phaseTimes,predLength);

% Function inputs: 
% - t2 = targets of selected surgery
% - yp1f = predictions of selected surgery (from section 4)
% - phaseTimes = phase times from data
% - predLength = length of selected surgery

% Outputs:
% Matrix for the percent classification of each phase, match counter, and
% number of mismatches of each phase.

% Compute overall clas acc stats
percentClass(isnan(percentClass)) = 0;
overallAcc = mean(percentClass);
stdAcc = std(percentClass);
% 





