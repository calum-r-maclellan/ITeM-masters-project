%% logRegStats.m - from 26.09 till 04.10
% Script to produce statistics on the logistic regression model, to get a
% better understanding of the data and suitability of this model for the 
% surgery time-prediction problem.
% Run this code in sections, since different sections perform different
% functions.
% TASKS
% -------------------------
% Task 1: 
% We want to see how the weights (w) are affected when rearranging the
% input data (ie the phase times).
% Task 2:
% Reduce the input space to see how the weights are affected when using
% less inputs.
% Task 3: 
% Calculate the predictive/classification accuracy of using an increasingly
% smaller input space. We expect the accuracy to decrease and follow a 
% cosine shape.  
% Task 4 (if theres time):
% Assess predictive/classification accuracy when changing the number of
% surgeries. 
clear; close all; 

%% Manual data analysis
% Let me control the number of surgeries and phases to help understand
% whats going on with the weights when parameters are changed.
load data.txt
nSurg = input('Number of surgeries: \n');
nPhases = input('Number of phases: \n'); 
x = data(1:nSurg,1:8);
    
% Compute weights and probabilities for each surgery
n = ones(nSurg,1);
w = glmfit(x(1:nSurg,1:nPhases),[x(1:nSurg,8) n],'binomial','link','logit');
warning('off','all');
y = glmval(w,x(1:nSurg,1:nPhases),'logit');
% Now compute accuracy using confusion matrix format 
a = 0; % a is n times prediction(1) = target(1) 
b = 0; % b is n times prediction(1) = target(0)
c = 0; % c is n times prediction(0) = target(1)
d = 0; % b is n times prediction(0) = target(0)
for ii = 1:nSurg
    if y(ii) > 0.5 && x(ii,8) > 0.5
       a=a+1;
    elseif y(ii) > 0.5 && x(ii,8) < 0.5
       b=b+1;
    elseif y(ii) < 0.5 && x(ii,8) > 0.5
       c=c+1;
    elseif y(ii) < 0.5 && x(ii,8) < 0.5
       d=d+1;
    end
end

% Model accuracy
acc = (a+d)/(a+b+c+d)
ConfusionMatrix = [NaN 1 0; 1 a b; 0 c d]

% ---------------------------------------
% Conclusion:
% By changing the number of surgeries, and the number of phases included,
% its clear that the accuracy and weights are affected by these two
% parameters. It seems that the weights decrease for lower number of
% surgeries considered, and that the accuracy decreases when less phases
% are used. However, 100% accuracy always occurs when all 7 phases are used
% - it only decreases as a function of phase number.
% In addition, randomising surgeries appeared to make no difference on
% overall results or weight values === WRONG!!!
% Now I need to code a for loop to iterate over a series of numbers to
% produce some stats on this. 


%% Compute stats for weights when randomising 60 surgeries 60 times
n = ones(60,1);
for r = 1: 60
    [m,~] = size(x); 
    rndise = randperm(m);
    X(rndise,1:8) = x(:,1:8);
    Xr = X(1:60,:);
    wX(:,r) = glmfit(Xr(:,1:7),[Xr(:,8) n],'binomial','link','logit');
end

for ii = 1: 7
    meanwX(ii) = mean(wX(ii+1,:));
    stdwX(ii)  = std(wX(ii+1,:));
end   
figure;
ax = axes;
bar(meanwX);
ax.YGrid = 'on';
ax.GridLineStyle = '-';
xticks(ax,1);
xticklabels(ax,{'\beta_1'});
title('Regression coefficients for phase 1 only');
xlabel('\bfCoefficients');
ylabel('\bfParameter Value');
hold on;
ngroups = size(meanwX,2);
nbars = size(meanwX,1);
groupwidth = min(0.8,nbars/(nbars+1.5));
x = (1:ngroups) - groupwidth/2 + (2-1)*groupwidth/(2*nbars);
errorbar(x, meanwX(1,:), stdwX(1,:),'rx','linestyle','none');

% Compute output   
% yx = glmval(wx,xr(:,1:7),'logit');
% yX = glmval(wX,Xr(:,1:7),'logit');
% % --------------------

%% Logistic regression statistics
% ==========================
% 1. Phase threshold calculation
% ==========================
% Create plot to show influence of each phase time on the predictive power.
% Segment every phase (column), reorder the times, and calculate the 
% predictions for that particular phase. This will tell us the times at
% which each phase can be before it results in a 1 (ie the times that a new
% surgery can have).
x = data(:,1:8);
t = sort(x(:,8));
n = ones(80,1);
xS = zeros(80,7); yS = zeros(80,7);
clf; figure(1);
for k = 1:7
    w = glmfit(x(:,k:k),[x(:,8) n],'binomial','link','logit');
    warning('off','all');
    y = glmval(w,x(:,k:k),'logit');
    xS(:,k)=sort(x(:,k)); yS(:,k)=sort(y);
    subplot(3,3,k);
    plot(xS(:,k),t(:,1),'ro',xS(:,k),yS(:,k),'b-','linewidth',1);
    title(['Phase ' num2str(k)]);
    xlabel('Time [s]');
    if k == 4
       ylabel('Probability');
    end
end

%%
% =======================================================
% 2. Changing the number of surgeries, keeping all phases
% =======================================================
% This loop calculates the regression coefficients (ie weights) for an
% increasing number of surgeries. It starts by calculating the weights for
% surgeries 1:8, storing them in the first column, then repeating for
% 1:16, 1:24, until 1:80. Each row corresponds to the weights in each phase,
% and each column is a block of surgeries.
% This will allow me to calculate meanpmstd for each of the seven phases.

% Note: the first row is the intercept coefficients (beta0), and so the 
% phase predictors are rows 2:8.

n = ones(80,1);
increment = 8; % iterate every 8 surgeries: gives 10 groups
W = [];
for j = 9:increment:80+1
    w = glmfit(x(1:j-1,1:7),[x(1:j-1,8) n(1:j-1)],'binomial','link','logit');
    warning('off','all');
    W = [W w];
end
weights = W(2:8,:);
meanWeights = zeros(1,7); stdWeights = zeros(1,7);
for j = 1:7
    meanWeights(j) = mean(weights(j,:));
    stdWeights(j)  = std(weights(j,:));
end
names = {'0','\beta_1','\beta_2','\beta_3','\beta_4','\beta_5','\beta_6','\beta_7'};
figure(2);
clf;
title('Regression coefficients for changing dataset size: all phases');
xlabel('\bfCoefficients');
ylabel('\bfParameter Value');
hold on;
    bar(meanWeights,'BarWidth',0.65);
    set(gca,'xticklabel',names);
    errorbar(meanWeights,stdWeights,'x');
hold off;

% Show how the weights change from each group of 8 surgeries
% Rather than just calculating the (mean pm std) of the data, this plot shows what happens to the weights when
% the dataset is changed. The variable W contains the weights for each phase 
% calculated for 10 groups of 8 surgeries. So, the first column are the weights for 
% surgeries 1:8, the second column are the weights for surgeries 1:16, and
% so on until 1:80.
% In effect, it shows what happens to the weights for an increasing number of surgical data.
% It also identifies which surgeries are causing the largest deviations. 
 
figure(3); 
clf;
hold all;
for i = 1:7
    plot(W(i+1,:),'-x','Linewidth',1);
end
labels = {'8','16','24','32','40','48','56','64','72','80'};
set(gca,'xticklabel',labels);
lgd2 = legend('\beta_1','\beta_2','\beta_3','\beta_4','\beta_5','\beta_6','\beta_7','location','best');
%title('Effect of increasing input size on regression coefficients');
xlabel('\bfNumber of surgeries included','fontsize',14);
ylabel('\bfParameter Value','fontsize',14);
lgd2.FontSize = 10;
% =======Conclusions ===================
% The results show that the weights eventuallz converge to a stable value
% as the input data size is increased. Initially, for the first 8 surgeries,
% large uncertainty is present in the weights, particularly for phases 1, 3, 5 and 6

%%
% ====================================================
% 3. Changing the actual surgeries, keeping the phases
% ====================================================
% Now compute stats for when different surgical data is used. Scan through
% all 80 videos, and compute the weights for 8 different blocks of 10
% surgeries. Start by calculating the weights for surgeries 1:10, then
% 11:20,..., till 71:80. This will give us an idea of how different data
% might affect the regression coefficients.
n = ones(80,1);
inc = 8; % increment by 10 surgeries
k = 0;
W1 = [];
for j = 1:inc:80
    k = k + 8;
    w = glmfit(x(j:k,1:7),[x(j:k,8) n(j:k)],'binomial','link','logit');
    warning('off','all');
    W1 = [W1 w];
end
weights1 = W1(2:8,:);
meanWeights1 = zeros(1,7); stdWeights1 = zeros(1,7);
for j = 1:7
    meanWeights1(j) = mean(weights1(j,:));
    stdWeights1(j)  = std(weights1(j,:));
end
names = {'0','\beta_1','\beta_2','\beta_3','\beta_4','\beta_5','\beta_6','\beta_7'};
figure(4);
clf;
title('Regression coefficients for different datasets: all phases');
xlabel('\bfCoefficients');
ylabel('\bfParameter Value');
hold on;
    bar(meanWeights1,'BarWidth',0.65);
    set(gca,'xticklabel',names);
    errorbar(meanWeights1,stdWeights1,'x');
hold off;

% Show how the weights change from each group of 8 surgeries
% Rather than just calculating the (mean pm std) of the data, this plot shows what happens to the weights when
% the dataset is changed. The variable W1 contains the weights for each phase 
% calculated for 10 groups of 8 surgeries. So, the first column are the weights for 
% surgeries 1:8, the second column are the weights for surgeries 9:16, and so on.
% This will give an idea of where the variability of the data lies. 
figure(5); 
clf;
hold all;
for i = 1:7
    plot(W1(i+1,:),'-x','Linewidth',1);
end
labels = {'1:8','9:16','17:24','25:32','33:40','41:48','49:56','57:64','65:72','73:80'};
set(gca,'xticklabel',labels);
lgd = legend('\beta_1','\beta_2','\beta_3','\beta_4','\beta_5','\beta_6','\beta_7','location','best');
%title('Effect of different datasets on regression coefficients');
xlabel('\bfSurgeries used','fontsize',14);
ylabel('\bfParameter Value','fontsize',14);
lgd.FontSize=10;
%%
% ====================================================
% 4. Changing surgery number (80), change number of phases
% ====================================================
% Now we want to see what happens to the value of the weights when we
% change the number of phases included in the data AND the surgeries. 
% Reason for changing both is that the entire group cant be grouped together 
% since different number of phases has different number of weights.
% So, here there will be 7 bars (one for each phase) that will have different weights produced
% number by the different number of surgeries.
n = ones(80,1);
for j = 80:-1:1
    for k = 7:-1:1
        w = glmfit(x(1:j,1:k),[x(1:j,8) n(1:j)],'binomial','link','logit');
        warning('off','all');
        if k == 7
           w7(:,j) = w;
            elseif k == 6
                   w6(:,j) = w;
                elseif k == 5
                       w5(:,j) = w;
                    elseif k == 4
                           w4(:,j) = w;
                    elseif k == 3
                           w3(:,j) = w;
                elseif k == 2
                       w2(:,j) = w;
            elseif k == 1
                   w1(:,j) = w;  
        end
    end
end

%% Plots: expect 7 different bar graphs; 1 for each reduction in number of
% phases from 7 to 1

% Weights for only phase 1
meanw1 = mean(w1(2,:)); 
stdw1  = std(w1(2,:));
figure(6);
clf;
ax = axes;
bar(meanw1);
ax.YGrid = 'on';
ax.GridLineStyle = '-';
xticks(ax,1);
xticklabels(ax,{'\beta_1'});
title('Regression coefficients for phase 1 only');
xlabel('\bfCoefficients');
ylabel('\bfParameter Value');
hold on;
ngroups = size(meanw1,2);
nbars = size(meanw1,1);
groupwidth = min(0.8,nbars/(nbars+1.5));
x = (1:ngroups) - groupwidth/2 + (2-1)*groupwidth/(2*nbars);
errorbar(x, meanw1(1,:), stdw1(1,:),'rx','linestyle','none');

% Weights for phase 1, 2 
for i = 1:2
    meanw2(i) = mean(w2(i+1,:));
    stdw2(i)  = std(w2(i+1,:));
end
figure(7);
clf;
ax = axes;
bar(meanw2);
ax.YGrid = 'on';
ax.GridLineStyle = '-';
xticks(ax,[1 2]);
xticklabels(ax,{'\beta_1','\beta_2'});
title('Regression coefficients for phase 1 and 2');
xlabel('\bfCoefficients');
ylabel('\bfParameter Value');
hold on;
ngroups = size(meanw2,2);
nbars = size(meanw2,1);
groupwidth = min(0.8,nbars/(nbars+1.5));
x = (1:ngroups) - groupwidth/2 + (2-1)*groupwidth/(2*nbars);
errorbar(x, meanw2(1,:), stdw2(1,:),'rx','linestyle','none');

% Weights for phase 1 to 3
for i = 1:3
    meanw3(i) = mean(w3(i+1,:));
    stdw3(i)  = std(w3(i+1,:));
end
figure(8);
clf;
ax = axes;
bar(meanw3);
ax.YGrid = 'on';
ax.GridLineStyle = '-';
xticks(ax,[1 2 3]);
xticklabels(ax,{'\beta_1','\beta_2','\beta_3',});
title('Regression coefficients for phase 1, 2 and 3');
xlabel('\bfCoefficients');
ylabel('\bfParameter Value');
hold on;
ngroups = size(meanw3,2);
nbars = size(meanw3,1);
groupwidth = min(0.8,nbars/(nbars+1.5));
x = (1:ngroups) - groupwidth/2 + (2-1)*groupwidth/(2*nbars);
errorbar(x, meanw3(1,:), stdw3(1,:),'rx','linestyle','none');


% Weights for phase 1 to 4 
for i = 1:4
    meanw4(i) = mean(w4(i+1,:));
    stdw4(i)  = std(w4(i+1,:));
end
figure(9);
clf;
ax = axes;
bar(meanw4);
ax.YGrid = 'on';
ax.GridLineStyle = '-';
xticks(ax,[1 2 3 4]);
xticklabels(ax,{'\beta_1','\beta_2','\beta_3','\beta_4'});
title('Regression coefficients for phases 1 to 4');
xlabel('\bfCoefficients');
ylabel('\bfParameter Value');
hold on;
ngroups = size(meanw4,2);
nbars = size(meanw4,1);
groupwidth = min(0.8,nbars/(nbars+1.5));
x = (1:ngroups) - groupwidth/2 + (2-1)*groupwidth/(2*nbars);
errorbar(x, meanw4(1,:), stdw4(1,:),'rx','linestyle','none');

% Weights for phase 1 to 5 
for i = 1:5
    meanw5(i) = mean(w5(i+1,:));
    stdw5(i)  = std(w5(i+1,:));
end
figure(10);
clf;
ax = axes;
bar(meanw5);
ax.YGrid = 'on';
ax.GridLineStyle = '-';
xticks(ax,[1 2 3 4 5]);
xticklabels(ax,{'\beta_1','\beta_2','\beta_3','\beta_4','\beta_5'});
title('Regression coefficients for phases 1 to 5');
xlabel('\bfCoefficients');
ylabel('\bfParameter Value');
hold on;
ngroups = size(meanw5,2);
nbars = size(meanw5,1);
groupwidth = min(0.8,nbars/(nbars+1.5));
x = (1:ngroups) - groupwidth/2 + (2-1)*groupwidth/(2*nbars);
errorbar(x, meanw5(1,:), stdw5(1,:),'rx','linestyle','none');

% Weights for phase 1 to 6
for i = 1:6
    meanw6(i) = mean(w6(i+1,:));
    stdw6(i)  = std(w6(i+1,:));
end
figure(11);
clf;
ax = axes;
bar(meanw6);
ax.YGrid = 'on';
ax.GridLineStyle = '-';
xticks(ax,[1 2 3 4 5 6]);
xticklabels(ax,{'\beta_1','\beta_2','\beta_3','\beta_4','\beta_5','\beta_6'});
title('Regression coefficients for phases 1 to 6');
xlabel('\bfCoefficients');
ylabel('\bfParameter Value');
hold on;
ngroups = size(meanw6,2);
nbars = size(meanw6,1);
groupwidth = min(0.8,nbars/(nbars+1.5));
x = (1:ngroups) - groupwidth/2 + (2-1)*groupwidth/(2*nbars);
errorbar(x, meanw6(1,:), stdw6(1,:),'rx','linestyle','none');

% Weights for including all phases
for i = 1:7
    meanw7(i) = mean(w7(i+1,:));
    stdw7(i)  = std(w7(i+1,:));
end
figure(12);
clf;
ax = axes;
bar(meanw7);
ax.YGrid = 'on';
ax.GridLineStyle = '-';
xticks(ax,[1 2 3 4 5 6 7]);
xticklabels(ax,{'\beta_1','\beta_2','\beta_3','\beta_4','\beta_5','\beta_6','\beta_7'});
title('Regression coefficients for all phases');
xlabel('\bfCoefficients');
ylabel('\bfParameter Value');
hold on;
ngroups = size(meanw7,2);
nbars = size(meanw7,1);
groupwidth = min(0.8,nbars/(nbars+1.5));
x = (1:ngroups) - groupwidth/2 + (2-1)*groupwidth/(2*nbars);
errorbar(x, meanw7(1,:), stdw7(1,:),'rx','linestyle','none');

%% 
% ====================================================
% 5. Accuracy stats: reducing the phase dimension
% ====================================================
% This part calculates the classification accuracy for a changing number of
% phases. Expect to decrease with the reducing input space. Get mean and
% std from changing number of surgeries. It starts by computing the
% accuracy when using all 80 surgeries and 7 phases till 80
% surgeries and 1 phase. Then it goes to 79 surgeries and 7 phases till 79
% surgeries and 1 phase. Expect to get a matrix size 7x80 whereby the
% accuracies of using each phase are in the rows, and columns are for using
% 1 to n surgeries. 
% Eg: last column is the acc's for using surgeries 1:80 and varying phase
% dimensions, 2nd last is 1:79 surgeries, then 1:78 all the way until the
% first column which is for using just 1 surgery and increasing number of
% phases.
x = data(1:80,:);
n = ones(80,1);
A = [];
for j = 80:-1:1
    for k = 7:-1:1
        w = glmfit(x(1:j,1:k),[x(1:j,8) n(1:j)],'binomial','link','logit');
        warning('off','all');
        y = glmval(w,x(1:j,1:k),'logit');
        a = 0;  b = 0; c = 0; d = 0;
        for kk = 1:j
            if y(kk) > 0.5 && x(kk,8) > 0.5
               a=a+1;
            elseif y(kk) > 0.5 && x(kk,8) < 0.5
               b=b+1;
            elseif y(kk) < 0.5 && x(kk,8) > 0.5
               c=c+1;
            elseif y(kk) < 0.5 && x(kk,8) < 0.5
               d=d+1;
            end
        end
        acc = (a+d)/(a+b+c+d);
        A(k,j) = acc; % classification accuracy
    end
end

%% Calculate stats and plot results
figure;
clf;
for p = 1:7
    meanA(p) = mean(A(p,:));
    stdA(p)  = std(A(p,:));
end
phases = [1 2 3 4 5 6 7];
plot(phases,meanA,'b-');
hold on;
errorbar(phases,meanA,stdA,'rx');
title('Classification accuracy for changing input size');
xlabel('\bfNumber of phases included');
ylabel('\bfAccuracy');
axis([.75 7.25,0.5 1.05]);

%% Classification accuracy when randomising data

n = ones(60,1);
for r = 1: 60
    [m,~] = size(x); 
    rndise = randperm(m);
    X(rndise,1:8) = x(:,1:8);
    Xr = X(1:60,:);
    wX = glmfit(Xr(:,1:7),[Xr(:,8) n],'binomial','link','logit');
end
%% NOTES
% What is strange is that when you lower the number of
% surgeries (as they were presented originally), the accuracy
% increases for some and decreases for other values. For
% example, evaluating just phase 7's predictive power on s=80 down to s=50 in 10s,
% the acc goes from 0.7125, 0.7, 0.7 to 0.74, which suggests that some
% surgeries give different predictions than others. This is expected, but
% what is strange is that you would thus expect by randomising the
% surgeries this would change the acc every time but it doesnt - its
% exactly the same!...


