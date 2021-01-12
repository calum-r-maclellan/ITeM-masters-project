% Train a support vector machine (svm) classifier 

clear; close all;
load data.txt

%% Split into training data: 60 vids
nTraining = input('Enter the number of surgeries for training: \n');
nTest     = 80 - nTraining;
nPhases = input('Enter the number of phases: \n');
x = data(1:nTraining,1:nPhases);
y = data(1:nTraining,8);
svmModel = fitcsvm(x,y);

CVsvmModel = crossval(svmModel);
classLoss  = kfoldLoss(CVsvmModel)

%% Test the trained svm classifier on new data
X = data(nTraining+1:80,1:nPhases);
[predictLabel,score] = predict(svmModel,X);

%% Now compute accuracy using confusion matrix 
a = 0; b = 0; c = 0; d = 0; 
actualLabel = data(nTraining+1:80,8);
misClasses = [];
for ii = 1:nTest
    if predictLabel(ii) == 1 && actualLabel(ii) == 1
       a = a+1;
    elseif predictLabel(ii) == 1 && actualLabel(ii) == 0
       b = b+1;
    elseif predictLabel(ii) == 0 && actualLabel(ii) == 1
       c = c+1;
    elseif predictLabel(ii) == 0 && actualLabel(ii) == 0
       d = d+1;
    end
    
     % Determine which surgeries were misclassified
    if predictLabel(ii) ~= actualLabel(ii)
       misClasses = [misClasses nTraining+ii];
    end
end

% Model accuracy
tp = a; % true positives (number of too long surgeries correctly identified as 1)
tn = d; % true negatives (number of on time/fast surgeries correctly identified as 0)
fp = b; % false positives (number of too long surgeries incorrectly identified as 0)
fn = c; % false negatives (number of on time surgeries incorrectly identified as 1)
sens = tp/(tp+fn); % proportion of ACTUAL too long surgeries that are identified as too long
spec = tn/(tn+fp); % proportion of ACTUAL on time surgeries that are identified as on time
acc = ((a+d)/(a+b+c+d))*100;
ConfusionMatrix = [NaN 1 0; 1 a b; 0 c d]
m = length(misClasses);

if isempty(misClasses)
   disp('No surgeries were misclassified.');
else
   fprintf('Classification accuracy: %4.2f\n\n',acc);
   fprintf('There were %d misclassified surgeries: \n',m);
   disp(misClasses);
end