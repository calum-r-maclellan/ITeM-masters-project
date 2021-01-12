% This script computes classification accuracy statistics for 6
% different models:
%   - Neural network (pattern recognition)
%   - Logistic regression
%   - Support vector machine (SVM)
%   - can try others too maybe (LDA,kNN,AdaBoost).

% Started: 10.10.17  Finished: 18.10.17

% Aim: 
% to select the best model for the time series predictions based on
% classification accuracy.

% Method:
% Evaluate the classification rate/accuracy of the algorithms by randomly
% dividing the 80 datasets (surgeries) and using 60 for training and the
% remaining 20 to make predictions. The accuracy of each model will then be
% computed for that specific organisation of training/testing datasets, and
% for an increasing input space size (ie acc for 1 phase to all 7 phases).
% This procedure is then repeated 80 times and classification accuracies
% are averaged and the std computed.
% A plot is then presented to illustrate the results. 
% In addition, the script is also capable of computing the predictive power
% that each specific phases has. 

% Sidenote:
% For fairness, the order that the data is currently presented in will be
% maintained for all computations. This way, each model is evaluated on the
% same data. 

%--------------------------
% ALTERNATIVELY!!!!!!!!!!!!!!
% USE THE CLASSIFICATION LEARNER APP:
% It lets you compare various machine learning algorithms, and to generate
% models from them.
% What I'll do is train a bunch of different algorithms, generate models of
% them, and assess their predictions. 
% Turns out that the results are very similar to my results from coding 
% the methods manually. BUT I have the neural network results too, 
% so I'm better.
%----------------------------

%% Load the data
clear;
load data.txt
x = data;
nTrain = 20;        % no of training surgeries
nTest = 80-nTrain;  % no of test videos 
n = ones(nTrain,1);

%% Neural network
% To properly format the data for a net, the inputs and targets need to be
% set up as cell arrays containing the individual observations. So, for
% surgery 1 its input will be a cell array of n phases size containing all
% the phase times. The target for this should also be a cell array, but
% essentially its just an array. 
% Because the number of elements in a cell array cant be changed
% iteratively, the loop changes x and assigns training/test data BEFORE it
% goes through the tonndata() function. 
% The network size changes iteratively, which is why the configuring
% parameters are within the loop. Otherwise errors that indicate a
% different input size being present than is expected occur relentlessly. 

x = data';
for p = 1: 7
    for r = 1: 80
        % Configure network parameters
        trainFcn = 'trainscg';  
        hiddenLayerSize = 10;
        net = patternnet(hiddenLayerSize, trainFcn);
        net.input.processFcns = {'removeconstantrows','mapminmax'};
        net.output.processFcns = {'removeconstantrows','mapminmax'};
        net.performFcn = 'crossentropy';  
        net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
                        'plotconfusion', 'plotroc'};
        % Randomise data. 
        [~,m]  = size(x); 
        rndise = randperm(m);
        X(:,rndise) = x;
        % Assign training and testing datasets.
        trainX = X(1:p,1:nTrain);    trainT = X(8,1:nTrain); 
        testX  = X(1:p,nTrain+1:80); testT  = X(8,nTrain+1:80);
        % Convert to nn format
        trainX = tonndata(trainX,true,false);  
        trainT = tonndata(trainT,true,false); 
        testX  = tonndata(testX,true,false); 
        testT  = tonndata(testT,true,false); 
        [net,tr] = train(net,trainX,trainT);
        y(:,r)   = net(testX);  
        t(:,r)   = testT;
        tp = 0;  fp = 0; fn = 0; tn = 0;
        for kk = 1: nTest
                if y{kk,r} > 0.75
                   yNet(kk,r) = 1;
                else
                   yNet(kk,r) = 0;
                end

                if yNet(kk,r) == 1 && testT{kk} == 1
                   tp = tp + 1;
               elseif yNet(kk,r) == 1 && testT{kk} == 0
                   fp = fp + 1;
               elseif yNet(kk,r) == 0 && testT{kk} == 1
                   fn = fn + 1;
               elseif yNet(kk,r) == 0 && testT{kk} == 0
                   tn = tn + 1;
                end
            nnSens = tp/(tp+fn); % model sensitivity
            nnSpec = tn/(tn+fp); % specificity
            nnacc = (tp+tn)/(tp+fp+fn+tn); % accuracy
            NnSens(p,r) = nnSens;
            NnSpec(p,r) = nnSpec;
            NnAcc(p,r) = nnacc;
            % sensitivity
            meanNnSens(p) = mean(NnSens(p,:));
            % specficity
            meanNnSpec(p) = mean(NnSpec(p,:));
            % accuracy
            meanNnAcc(p) = mean(NnAcc(p,:));
        end
    end
end
averageNnSens = mean(meanNnSens);
averageNnSpec = mean(meanNnSpec);
averageNnAcc = mean(meanNnAcc);

stdNnSens = std(meanNnSens);
stdNnSpec = std(meanNnSpec);
stdNnAcc = std(meanNnAcc);

%% Logistic regression
% Classification acc when randomising data
% Method: 
% This pieice of code trains a logisitic regression model
% on 60 surgeries and tests on 20. But, because some surgeries are harder
% to classify, the training and testing datasets were randomised r=80 times
% and the output (y) computed each time. From this, the mean and std are
% calculated. This procedure is then repeated 7 times to incorporate the
% effect of different numbers of phases on the accuracy.
% IOW: 
% --> first iteration -> randomise data -> assign 60 training and 20 test 
% --> compute weights -> compute output -> compute accuracy.
% --> 2nd iteration...repeat...3rd iteration...till 80th iteration. 
% With this, the accuracy is computed according to a CROSS-VALIDATION strategy
% to see if the accuracy changes when different datasets are used for
% training. This is expected since the decision boundaries will be
% different for different surgeries, and the model will learn certain
% dependencies from certain surgeries and not from others. 

% NB 1: The first 60 randomised vids are training, the last 20 are for test. 
% NB 2: To obtain the plot for classification accuracy for each phase,
% change the 1:p in line ... and ... to p:p and uncomment appropriate plot
% labels. This switches calculations to each phase rather than an
% increasing 1:p=7 number of phases. 
clear; 
load data.txt
x = data;
nTrain = 60;        % no of training surgeries
nTest = 80-nTrain;  % no of test videos 
n = ones(nTrain,1);
for p = 1: 7
    for r = 1: 80
        [m,~] = size(x); 
        rndise = randperm(m);
        X(rndise,1:8) = x(:,1:8);
        trainX = X(1:nTrain,1:p);   trainT = X(1:nTrain,8);
        testX = X(nTrain+1:80,1:p); testT = X(nTrain+1:80,8);
        w = glmfit(trainX,[trainT n],'binomial','link','logit');
        warning('off','all');
        y(:,r) = glmval(w,testX,'logit');
        t(:,r) = testT;
        tp = 0; fp = 0; fn = 0; tn = 0;
        for ii = 1: nTest
            if y(ii,r) > 0.5
               yLog(ii,r) = 1;
            else
               yLog(ii,r) = 0;
            end
            % Conf matrix 
            if yLog(ii,r) == 1 && testT(ii) == 1
               tp = tp + 1;
            elseif yLog(ii,r) == 1 && testT(ii) == 0
               fp = fp + 1;
            elseif yLog(ii,r) == 0 && testT(ii) == 1
               fn = fn + 1;
            elseif yLog(ii,r) == 0 && testT(ii) == 0
               tn = tn + 1;
            end
        end
        lrSens = tp/(tp+fn); % model sensitivity
        lrSpec = tn/(tn+fp); % specificity
        lracc = (tp+tn)/(tp+fp+fn+tn); % overall accuracy
        LRSens(p,r) = lrSens;
        LRSpec(p,r) = lrSpec;
        LRAcc(p,r) = lracc;
        % sensitivity
        meanLrSens(p) = mean(LRSens(p,:));
        % specficity
        meanLrSpec(p) = mean(LRSpec(p,:));
        % accuracy
        meanLrrAcc(p) = mean(LRAcc(p,:));
    end
end
averageLrSens = mean(meanLrSens);
averageLrSpec = mean(meanLrSpec);
averageLrAcc = mean(meanLrrAcc);

stdLrSens = std(meanLrSens);
stdLrSpec = std(meanLrSpec);
stdLrAcc = std(meanLrrAcc);

%% SVM
% Build a svm model to carry out the phase time predictions. 
% Also computes accuracy, specificity and sensitivity.
clear;
load data.txt
x = data;
nTrain = 60;        % no of training surgeries
nTest = 80-nTrain;  % no of test videos 
n = ones(nTrain,1); 
for p = 1: 7
    for r = 1: 80
        [m,~]  = size(x); 
        rndise = randperm(m);
        X(rndise,1:8) = x;
        trainX = X(1:nTrain,1:p);   trainT = X(1:nTrain,8);
        testX  = X(nTrain+1:80,1:p); testT = X(nTrain+1:80,8);
        svmModel = fitcsvm(trainX,trainT);
        [y(:,r),score] = predict(svmModel,testX);
        t(:,r)   = testT;
        tp = 0; fp = 0; fn = 0; tn = 0;
        for ii = 1: nTest
            if y(ii,r) > 0.5
               ySvm(ii,r) = 1;
            else
               ySvm(ii,r) = 0;
            end
            % Conf matrix 
            if ySvm(ii,r) == 1 && testT(ii) == 1
               tp = tp + 1;
            elseif ySvm(ii,r) == 1 && testT(ii) == 0
               fp = fp + 1;
            elseif ySvm(ii,r) == 0 && testT(ii) == 1
               fn = fn + 1;
            elseif ySvm(ii,r) == 0 && testT(ii) == 0
               tn = tn + 1;
            end
        end
        svmSens = tp/(tp+fn); % model sensitivity
        svmSpec = tn/(tn+fp); % specificity
        svmacc = (tp+tn)/(tp+fp+fn+tn);
        SvmSens(p,r) = svmSens;
        SvmSpec(p,r) = svmSpec;
        SvmAcc(p,r) = svmacc;
        % sensitivity
        meanSvmSens(p) = mean(SvmSens(p,:));
        % specficity
        meanSvmSpec(p) = mean(SvmSpec(p,:));
        % accuracy
        meanSvmAcc(p) = mean(SvmAcc(p,:));
    end
end
averageSvmSens = mean(meanSvmSens);
averageSvmSpec = mean(meanSvmSpec);
averageSvmAcc = mean(meanSvmAcc);

stdSvmSens = std(meanSvmSens);
stdSvmSpec = std(meanSvmSpec);
stdSvmAcc = std(meanSvmAcc);

%% LDA
clear;
load data.txt
x = data;
nTrain = 60;        % no of training surgeries
nTest = 80-nTrain;  % no of test videos 
n = ones(nTrain,1); 
for p = 1: 7
    for r = 1: 80
        [m,~]  = size(x); 
        rndise = randperm(m);
        X(rndise,1:8) = x;
        trainX = X(1:nTrain,1:p);   trainT = X(1:nTrain,8);
        testX  = X(nTrain+1:80,1:p); testT = X(nTrain+1:80,8);
        ldaModel = fitcdiscr(trainX,trainT);
        [y(:,r),score] = predict(ldaModel,testX);
        t(:,r)   = testT;
        tp = 0; fp = 0; fn = 0; tn = 0;
        for ii = 1: nTest
            if y(ii,r) > 0.5
               yLda(ii,r) = 1;
            else
               yLda(ii,r) = 0;
            end
            % Conf matrix 
            if yLda(ii,r) == 1 && testT(ii) == 1
               tp = tp + 1;
            elseif yLda(ii,r) == 1 && testT(ii) == 0
               fp = fp + 1;
            elseif yLda(ii,r) == 0 && testT(ii) == 1
               fn = fn + 1;
            elseif yLda(ii,r) == 0 && testT(ii) == 0
               tn = tn + 1;
            end
        end
        ldaSens = tp/(tp+fn); % model sensitivity
        ldaSpec = tn/(tn+fp); % specificity
        ldaacc = (tp+tn)/(tp+fp+fn+tn);
        LdaSens(p,r) = ldaSens;
        LdaSpec(p,r) = ldaSpec;
        LdaAcc(p,r)  = ldaacc;
        % sensitivity
        meanLdaSens(p) = mean(LdaSens(p,:));
        % specficity
        meanLdaSpec(p) = mean(LdaSpec(p,:));
        % accuracy
        meanLdaAcc(p) = mean(LdaAcc(p,:));
    end
end
averageLdaSens = mean(meanLdaSens);
averageLdaSpec = mean(meanLdaSpec);
averageLdaAcc  = mean(meanLdaAcc);

stdLdaSens = std(meanLdaSens);
stdLdaSpec = std(meanLdaSpec);
stdLdaAcc  = std(meanLdaAcc);

%% kNN
clear;
load data.txt
x = data;
nTrain = 60;        % no of training surgeries
nTest = 80-nTrain;  % no of test videos 
n = ones(nTrain,1);
for p = 1: 7
    for r = 1: 80
        [m,~]  = size(x); 
        rndise = randperm(m);
        X(rndise,1:8) = x;
        trainX = X(1:nTrain,1:p);   trainT = X(1:nTrain,8);
        testX  = X(nTrain+1:80,1:p); testT = X(nTrain+1:80,8);
        knnModel = fitcknn(trainX,trainT);
        [y(:,r),score] = predict(knnModel,testX);
        t(:,r)   = testT;
        tp = 0; fp = 0; fn = 0; tn = 0;
        for ii = 1: nTest
            if y(ii,r) > 0.5
               yKnn(ii,r) = 1;
            else
               yKnn(ii,r) = 0;
            end
            % Conf matrix 
            if yKnn(ii,r) == 1 && testT(ii) == 1
               tp = tp + 1;
            elseif yKnn(ii,r) == 1 && testT(ii) == 0
               fp = fp + 1;
            elseif yKnn(ii,r) == 0 && testT(ii) == 1
               fn = fn + 1;
            elseif yKnn(ii,r) == 0 && testT(ii) == 0
               tn = tn + 1;
            end
        end
        knnSens = tp/(tp+fn); % model sensitivity
        knnSpec = tn/(tn+fp); % specificity
        knnacc  = (tp+tn)/(tp+fp+fn+tn);
        KnnSens(p,r) = knnSens;
        KnnSpec(p,r) = knnSpec;
        KnnAcc(p,r)  = knnacc;
        % sensitivity
        meanKnnSens(p) = mean(KnnSens(p,:));
        % specficity
        meanKnnSpec(p) = mean(KnnSpec(p,:));
        % accuracy
        meanKnnAcc(p) = mean(KnnAcc(p,:));
    end
end
averageKnnSens = mean(meanKnnSens);
averageKnnSpec = mean(meanKnnSpec);
averageKnnAcc  = mean(meanKnnAcc);

stdKnnSens = std(meanKnnSens);
stdKnnSpec = std(meanKnnSpec);
stdKnnAcc  = std(meanKnnAcc);

%% AdaBoost 
clear;
load data.txt
x = data;
nTrain = 60;        % no of training surgeries
nTest = 80-nTrain;  % no of test videos 
n = ones(nTrain,1);
for p = 1: 7
    for r = 1: 80
        [m,~]  = size(x); 
        rndise = randperm(m);
        X(rndise,1:8) = x;
        trainX = X(1:nTrain,1:p);   trainT = X(1:nTrain,8);
        testX  = X(nTrain+1:80,1:p); testT = X(nTrain+1:80,8);
        adaModel = fitensemble(trainX,trainT,'AdaBoostM1',100,'Tree');
        [y(:,r),score] = predict(adaModel,testX);
        t(:,r)   = testT;
        tp = 0; fp = 0; fn = 0; tn = 0;
        for ii = 1: nTest
            if y(ii,r) > 0.5
               yAda(ii,r) = 1;
            else
               yAda(ii,r) = 0;
            end
            % Conf matrix 
            if yAda(ii,r) == 1 && testT(ii) == 1
               tp = tp + 1; % how many times was it correctly classified as 1
            elseif yAda(ii,r) == 1 && testT(ii) == 0
               fp = fp + 1; % how many times was it classified as 1 when it should of been 0
            elseif yAda(ii,r) == 0 && testT(ii) == 1
               fn = fn + 1; % how many times was it classified as 0 when it should of been 1
            elseif yAda(ii,r) == 0 && testT(ii) == 0
               tn = tn + 1; % how many times was it correctly classified as 0
            end
        end
        adaSens = tp/(tp+fn); % model sensitivity
        adaSpec = tn/(tn+fp); % specificity
        adaacc = (tp+tn)/(tp+fp+fn+tn);
        AdaSens(p,r) = adaSens;
        AdaSpec(p,r) = adaSpec;
        AdaAcc(p,r)  = adaacc;
        % sensitivity
        meanAdaSens(p) = mean(AdaSens(p,:));
        % specficity
        meanAdaSpec(p) = mean(AdaSpec(p,:));
        % accuracy
        meanAdaAcc(p) = mean(AdaAcc(p,:));
    end
end
averageAdaSens = mean(meanAdaSens);
averageAdaSpec = mean(meanAdaSpec);
averageAdaAcc  = mean(meanAdaAcc);

stdAdaSens = std(meanAdaSens);
stdAdaSpec = std(meanAdaSpec);
stdAdaAcc  = std(meanAdaAcc);


% set(gca,'FontSize',16);

%% Ran each section and got the following results for each:

% Logistic regression
meanLr = [71.8 81.4 82.4 94.1 93.3 93.8 92.8]; 
% stdLr  = [8.89  6.99  8.10  5.89  6.11  4.87  5.44 ];
% lrSens = 100;
% lrSpec = 100;
figure(1);
axis([.5 7.5,0.5 1.05]);
plot(meanLr,'-o','LineWidth',2);
xlabel('\bfNumber of phases included','fontsize',14);
ylabel('\bfClassification Accuracy (%)','fontsize',14);

%%
% Neural network
meanNn = [70.1 78.6 77.0 85.7 85.5 85.2 84.5];

%% SVM
meanSvm = [73.1 81.9 81.4 94.3 92.1 95.8 94.4];


%% LDA
meanLda = [72.4 80.8 82.4 88.7 87.5 90.8 90.9];


%% KNN
meanKnn = [52.9 72.4 73.8 88.3 89.0 89.3 88.6];
 

%% AdaBoost
meanAda = [64.9 76.8 76.1 86.2 85.0 87.4 87.4];

%% Plot all results
% This graph will help visualise the classification rate results of each
% algorithm, and to determine the best algorithm to select for the
% predictive model in later studies. 

figure(2);
phases = [1 2 3 4 5 6 7];
plot(phases,meanLr,'-p','Linewidth',2);
hold all;
plot(phases,meanNn,'--h','Linewidth',2);
plot(phases,meanSvm,'-d','Linewidth',2);
plot(phases,meanLda,'-s','Linewidth',2);
plot(phases,meanKnn,'-x','Linewidth',2);
plot(phases,meanAda,':^','Linewidth',2);
% title('Classification accuracy for various algorithms','fontsize',14);
xlabel('\bfNumber of phases included','fontsize',14);
ylabel('\bfClassification Accuracy (%)','fontsize',14);
lgd = legend('Logistic Regression','Neural Network','SVM','LDA','KNN',...
       'AdaBoost','Location','southeast');
lgd.FontSize = 10;

