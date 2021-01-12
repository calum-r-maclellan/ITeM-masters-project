%% Load data
clear;
load phaseSurgeryTime_data.txt

%% Prepare data
nTrain = input('Enter the number of videos that will be used to train the network: \n');
nTest = 80 - nTrain;
nPhases = input('Enter the number of phases: \n');
x = phaseSurgeryTime_data(:,1:nPhases)';
T_sur = phaseSurgeryTime_data(:,8)';
load data.txt
t = data(:,8)';
X = tonndata(x,true,false);    
T = tonndata(t,true,false);
trainX = X(:,1:nTrain); testX = X(:,nTrain+1:80);
trainT = T(1:nTrain);   testT = T(nTrain+1:80);

%%
% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize, trainFcn);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testingons
% 4 functions:
% - dividerand:
% - divideblock:
% - divideint:
% - divideind:

% % Divide data randomly
% net.divideFcn = 'dividerand';
% net.divideParam.trainRatio = 50/100;
% net.divideParam.valRatio= 25/100;
% net.divideParam.testRatio = 25/100;

% divide by indices (ie specifying which datasets are train,val,test)
% net.divideFcn = 'divideind'; 
% net.divideParam.trainInd = 1:40;
% net.divideParam.valInd = 41:60;
% net.divideParam.testInd = 61:80;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'crossentropy';  % Cross-Entropy

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotconfusion', 'plotroc'};

%%
% Train the Network
[net,tr] = train(net,trainX,trainT);

% Test the Network
y = net(testX);

%%
% Outputs are 0.5 or 1 - round values down or up.
for k = 1:nTest
    if y{k} > 0.75
       decision(k) = 1;
    else
       decision(k) = 0;
    end
end

%%
misClasses = [];
a=0; b=0; c=0; d=0;
for ii = 1:nTest
    if decision(ii) == 1 && testT{ii} == 1
       a = a + 1;
    elseif decision(ii) == 1 && testT{ii} == 0
       b = b + 1;
    elseif decision(ii) == 0 && testT{ii} == 1
       c = c + 1;
    elseif decision(ii) == 0 && testT{ii} == 0
       d = d + 1;
    end
    
    % Determine which surgeries were misclassified
    if decision(ii) ~= testT{ii}
       misClasses = [misClasses nTrain+ii];
    end
end

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

%%
% e = gsubtract(T_sur,y);
% performance = perform(net,T_sur,y);
% tind = vec2ind(T_sur);
% yind = vec2ind(y);
% percentErrors = sum(tind ~= yind)/numel(tind);
% 
% % Recalculate Training, Validation and Test Performance
% trainTargets = T_sur .* tr.trainMask{1};
% valTargets = T_sur .* tr.valMask{1};
% testTargets = T_sur .* tr.testMask{1};
% trainPerformance = perform(net,trainTargets,y);
% valPerformance = perform(net,valTargets,y);
% testPerformance = perform(net,testTargets,y);
% 
% % View the Network
% view(net)

%% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotconfusion(t,y)
%figure, plotroc(t,y)

