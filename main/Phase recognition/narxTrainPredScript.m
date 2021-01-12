%% Perform real-time phase recognition using a narx network.
% Trains a NARX neural network using a user-defined number of surgeries,
% and then allows the user to select a surgery from the test set for making
% predictions.
% Version 2.1.
% Started: 23.10.17. Finished: 27.10.17

% Initialise
clear; close all; 


%% Section 1. ==============
% Run Cholec80_Main.m first to get necessary variables
run Cholec80_Main.m 

%% Remove unecessary variables - keep tool signals, targets, and surgery times
clearvars -except toolInputs phaseTargets T_sur T_ph 

%% Format narx data
% Ask user for number of training videos  
nTrain = input('Enter the number of surgeries to train the NARX with: \n');
nTest = 80-nTrain;

% Take variables and convert data to useful format. 
[u_tools, y_phases, u, y] = narxdatafcn(toolInputs,phaseTargets,T_sur,nTrain);


%% Section 2.
% Train the NARX network using a series-parallel architecture
% Configure network parameters

id = 1; % input delays
fd = 1; % feedback delays
hiddenNeurons = 10; % number of neurons in the hidden layer
narx_net = narxnet(1:id,1:fd,hiddenNeurons); % create narx
narx_net.trainFcn = 'trainbr';
narx_net.performFcn = 'sse';
narx_net.layers{1}.transferFcn = 'tansig';
narx_net.layers{2}.transferFcn = 'purelin';
narx_net.divideFcn = 'dividetrain';  
narnx_net.trainParam.epochs = 1000;
narx_net.trainParam.min_grad = 1e-6;
[p,Pi,Ai,t] = preparets(narx_net,u_tools,{},y_phases);

% Compute training time
tic
[narx_net,tr] = train(narx_net,p,t,Pi);
telapsed = toc;
trainTime = floor(telapsed/60) + ((telapsed/60)-floor(telapsed/60))*60/100; 

% Now train the network 
y_train = narx_net(p,Pi,Ai);

%% Calculate the error = target - prediction 
eTrain = gsubtract(t,y_train);
performance = perform(narx_net,t,y_train)


%% autocorr plots
ploterrcorr(eTrain);
xlabel('Lag');
ylabel('Correlation');
set(gca,'FontSize',12);


%% inerrcorr plots
plotinerrcorr(p,eTrain);
xlabel('Lag');
ylabel('Correlation');
set(gca,'FontSize',12);

%% Plot network TRAINING results: Uncomment desired plots
% figure, plotperform(tr);
% figure, plottrainstate(tr);
% figure, plotregression(t,y_train)
% figure, plotresponse(t,y_train)

% Plot training predictions against ground truth
% Allow user to choose training surgery to see
s = input(['Select one of the ' num2str(nTrain) ' training surgeries to view: \n']);

if s > nTrain
   fprintf('Please choose a surgery between 1 and %i\n',nTrain);
   s = input(['Select one of the ' num2str(nTrain) ' training surgeries to view: \n']);
end

for i = 1: length(y_train)
    y2(i) = y_train{1,i}(s);
    tplot(i) = t{1,i}(s);
end

y2(isnan(y2)) = []; 
tplot(isnan(tplot)) = [];
yplot = trainfilt(y2,length(y2));
yplot = yplot(1:length(yplot)-1);
time = 1: 1: length(yplot);
figure(3);
clf;
plot(time,tplot,'b','LineWidth',2);
hold on;
plot(time,yplot,'r','linewidth',1);
title('NARX network outputs for training samples');
xlabel('Time [s]');
ylabel('Phase Number');
legend('Ground truth','Predicted','location','best');
axis([0 length(time),0.5 7.5]);


%% Section 3.
% Use trained network for making predictions.
% This is the part where targets are removed, the open-loop is closed,
% and the network is allowed to prove its predictive ability on new data. 

% Ask user which surgery to use for testing
dc = nTrain+1;
md = 80;
surg = input(['Enter the surgery to be predicted (' num2str(dc) ' to ' num2str(md) '): \n']);
narx_netc = closeloop(narx_net); % create parallel architecture
view(narx_netc)
[p1,Pi1,Ai1,t1] = preparets(narx_netc,u{surg},{},y{surg});
yp1 = narx_netc(p1,Pi1,Ai1);
TS = size(t1,2);

% Compute and plot error stats
e = zeros(1,TS);
target = cell2mat(t1);
for i = 1: TS
   e(i) = (target(i)-cell2mat(yp1(i)))^2;
end

sumSE = sum(e)
meanSE = sumSE/TS
figure(5); 
plot(e); 
title(['NARX network: surgery ' num2str(surg) ' prediction errors']);
xlabel('Time [s]');
ylabel('Error');

% Plot targets against raw outputs. 
figure(1);
subplot(2,1,1);
plot(1:TS,cell2mat(t1),'b',1:TS,cell2mat(yp1),'r');
title(['NARX network: surgery ' num2str(surg) ' phase predictions']);
xlabel('Time [s]');
ylabel('Phase Number');
lgd = legend('True','Predicted','Location','best');

%% Section 4.
% Apply heaviside function to level off plot
% For easier visualisation of predictions, implement heaviside method to
% level off spikes.
predLength = length(yp1); % get length of matrix of predicted values
yp1f = predfilt(yp1,predLength); % filter values

% Plot cleaned results
figure(1);
subplot(2,1,2);
plot(1:TS,cell2mat(t1),'b',1:TS,yp1f,'r');
xlabel('Time [s]');
ylabel('Phase Number');
lgd = legend('True','Predicted','Location','best');

%% Section 5: Computing the network accuracy
% This section involves taking the fitered outputs, yp1f, and computing the
% confusion matrix associated with the corresponding targets of each
% surgery. Of course, different surgeries will have different accuracies
% and therefore the overall accuracy (for the 10 vids) will be the mean of
% all the accuracies.

% load data.txt
% T_ph = data(:,1:7);

% Shift the first phase by nTdl timesteps, NOT all the phases. 
nTdl = fd; % number of tapped delay lines
phaseTimes = T_ph(surg,:);
phaseTimes(1) = phaseTimes(1)-nTdl; 
t2 = cell2mat(t1); % convert cl targets from cell to matrix
timeDelay = zeros(1,7);
[percentClass,phaseCount,phaseDiff,timeDelay] = narxClassStats(t2,yp1f,phaseTimes,predLength,timeDelay);

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


%% NOTES
%-------------------------------
% Previous formatting techniques:
%  Attempt 1:
%  Create tool inputs as a 7xn matrix, where n is the length of time of x 
%  surgeries, and feed directly into narx. Phases were coded as a 1xn
%  matrix of 1s/2s etc where phase times from T_ph were used to separate
%  the phases.
%  Didnt really work as this was taken as a continuous surgery, and 
%  tools/phases were hard to distinguish as being part of separate surgeries.

%  Attempt 2:
%  Same tool format, but phases were instead coded with a 1 on a 7xn matrix
%  so that 0s for all other phases except current one. Still continuous, so
%  didnt really work. Plus, the network performed very poorly when trying
%  to separate phases on new data.

%  Attempt 3:
%  Same idea as before, but used tonndata to convert each timestep signal
%  into a cell to be fed into the network. Coded phases with 1s-2s in
%  place of 1s-0s using same format as attempt 2. Didnt work, and was only
%  able to code for one surgery at a time. Couldnt train on one dataset and
%  test on another since surgeries were of different length and couldnt 
%  therefore be grouped into one matrix, so needed a way to attribute each 
%  signal to its phase of a particular surgery. 

%  FINAL ATTEMPT:
%  This way works. I used the tool format from before, except that I
%  created a 1x80 (80 surgeries) cell 'u' where each cell contained a cell of
%  1xm length (m=T_sur) of 7x1 matrices of 1s and 0s for tool
%  data. 
%  Phases 'y' were coded the same way, except that each 1xm cell contained the
%  phase number at their corresponding T_ph times. Still had the issue of
%  not being able to group surgeries together due to varying lengths...

%  Heres the trick: I concatenated all of the surgeries together at
%  each timestep for length l, where l is the longest surgery of the
%  dataset, by filling the empty arrays of shorter surgeries with
%  NaNs so that the network could process the useful info and ignore the
%  NaNs. Did the same with the phases. Since this was for the training
%  dataset, I ask the user for nTrain and concatenate nTrain surgeries
%  together and leave 80-nTrain for testing. 

%  Example: the longest surgery is 5993s long, so the tools 'u_tools 'and
%  phases 'y_phases' were 1x5993 cells. Ask user for nTrain=70 and concatenate
%  70 surgeries together at each timestep.
%  Cell 1 of u_tools (ie u_tools{1,1}) is a 7x70 double and each column 
%  contains the 1st timestep signal from that surgery (ie column 23 has a
%  7x1 matrix of 1s-0s of tool signals for surgery 23 at t= 1second).
%  Cell 2 contains the same info but for the t=2seconds, and so on until
%  5993. By this point, the longest surgery is complete and all other
%  surgeries have NaNs in place of [] empty cells. 
%  
%  The targets have the exact same format. 
%  Now what happens is the tools and targets are matched so that each
%  element of each timestep corresponds to the signal and target of each
%  surgery (7x1 tool matrix paired with phase number 1...7). 

