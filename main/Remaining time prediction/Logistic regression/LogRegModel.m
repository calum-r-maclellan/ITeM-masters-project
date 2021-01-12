%% Logistic regression model 
% Script to perform a simple logistic regression model on the phase data to
% produce probabilities of total surgery times (of each surgery) being too
% long, based on the phase times.

% Load duration data generated from Cholec80_Main.m.
% Assign phase durations to T_ph, and total surgery durations to T_sur
clear
load phaseSurgeryTime_data.txt
T_ph  = phaseSurgeryTime_data(:,1:7); 
T_sur = phaseSurgeryTime_data(:,8);
load data.txt
durTargets = data(:,8);

%% Train the model and test on new data
% Now that we know the model works (has 100% accuracy when evaluated on 
% entire dataset), do training and testing.
% Repeat entire process as before except with separate dataset sizes.

%-----------------------------------------------------%
% NOTE: when running this section, comment out the last  
%-----------------------------------------------------%

% Specify the number of surgeries for training and testing.
clear; close all;
load data.txt
x = data;
durTargets = x(:,8);
nTraining   = input('Number of training surgeries: \n'); % set the number of training surgeries...
nTesting    = 80-nTraining; % ...and corresponding test surgeries
nPhases = input('Enter the number of phases: \n');
trainData   = x(1:nTraining,1:nPhases); 
trainLabels = durTargets(1:nTraining,:);
testData    = x(nTraining+1:80,1:nPhases); 
testTargets = durTargets(nTraining+1:80);

% Train the model on 60 surgeries. Use generalised logistic model fit.
n = ones(length(trainData),1);
w1 = glmfit(trainData,[trainLabels n],'binomial','link','logit');
warning('off','all');
y = glmval(w1,testData,'logit');
% Calculate prediction coefficients for n (test) surgeries by hand.
% Z = testData*(w1(2:end));
% a = ones(length(Z),1);
% A = w1(1)*a;
% Z_new = Z+A;
% Output = Logistic(Z_new);

% Outputs are almost 0 and almost 1 - round values down or up.
for k = 1:nTesting
    if y(k) > 0.5
       y(k) = 1;
    else
       y(k) = 0;
    end
end

a = 0; b = 0; c = 0; d = 0; mcCount = 0;
% where,
% a is n times prediction(1) = target(1) 
% b is n times prediction(1) = target(0)
% c is n times prediction(0) = target(1)
% d is n times prediction(0) = target(0)

% Run through all the probabilities and match them with the targets to get
% an idea of the accuracy of the model. 
misClasses = [];

for ii = 1:nTesting
    if y(ii) > 0.5 && testTargets(ii) > 0.5
       a = a + 1;
    elseif y(ii) > 0.5 && testTargets(ii) < 0.5
       b = b + 1;
    elseif y(ii) < 0.5 && testTargets(ii) > 0.5
       c = c + 1;
    elseif y(ii) < 0.5 && testTargets(ii) < 0.5
       d = d + 1;
    end
    
    % Determine which surgeries were misclassified
    if y(ii) ~= testTargets(ii)
       misClasses = [misClasses nTraining+ii];
    end
end

% Compute confusion matrix by hand and display.
Acc = (a+d)/(a+b+c+d);
percentAcc = Acc*100;
ConfusionMatrix = [NaN 1 0; 1 a b; 0 c d] 
% where: rows=output class, columns=target class
m = length(misClasses);

if isempty(misClasses)
   disp('No surgeries were misclassified.');
else
   fprintf('Classification accuracy: %4.2f\n\n',percentAcc);
   fprintf('There were %d misclassified surgeries: \n',m);
   disp(misClasses);
end


%% Build a logistic regression model
% Having some issues with nn so decided to try a simpler model. Here a
% logistic model is hand-built to get the probabilties of each surgery,
% whether or not it is too long (1) or on time/faster (0). 
% This model is 
% Get coefficients directly and save to z_new
% N = ones(length(T_ph),1);
% w = glmfit(T_ph,[durTargets N],'binomial','link','logit'); % weights
% Z = T_ph*(w(2:end));
% a = ones(length(Z),1);
% A = w(1)*a;
% z_new = Z+A;
% 
% % compute the logistic function of the input coefficients
% Output = Logistic(z_new);
% 
% % Now compute accuracy using confusion matrix format (ie four corners)
% a = 0; % a is n times prediction(1) = target(1) 
% b = 0; % b is n times prediction(1) = target(0)
% c = 0; % c is n times prediction(0) = target(1)
% d = 0; % b is n times prediction(0) = target(0)
% 
% for ii = 1:80
%     if Output(ii) > 0.5 && durTargets(ii) > 0.5
%        a = a+1;
%     elseif Output(ii) > 0.5 && durTargets(ii) < 0.5
%        b=b+1;
%     elseif Output(ii) <= 0.5 && durTargets(ii) >= 0.5
%        c=c+1;
%     elseif Output(ii) <= 0.5 && durTargets(ii) <= 0.5
%        d=d+1;
%     end
% end
% 
% % Model accuracy
% acc = (a+d)/(a+b+c+d)
% ConfusionMatrix = [NaN 1 0; 1 a b; 0 c d]

%% Code to create the dataset containing phase times and targets (1/0)
% % Count number of datasets that exceed threshold

% % Set threshold
% targetTime = median(T_sur); % set median of T_sur as the target surgery duration
% th = 600; % ±10mins threshold
% counter = 0;
% durTargets = zeros(80,1); % initialise vector for storing targets for all 80 surgeries
% 
% % check training datasets (40) to see if any are outside of threshold. if
% % they are, assign a 1 to them, otherwise a 0. This will create a vector of
% % 1s and 0s that will be targets into a nn. Inputs will be the 
% for i = 1:80
%     if  T_sur(i) - targetTime >  th
%         counter = counter + 1;
%         durTargets(i,:) = 1;
%     else
%         durTargets(i,:) = 0;
%     end
% end

% Concatenate T_ph and durTargets so that I can randomise the data more,
% and test for the stats in logRegStats.m
% data = [T_ph durTargets];
% dlmwrite('data.txt',data,'delimiter','\t');

