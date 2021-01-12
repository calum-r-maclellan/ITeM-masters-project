%% 29/8/17 
% REVISION 1 

% Main that uses Cholec80_Main.m outputs and feed into neural network. 

%%
% Going to try something different. The previous results I got took into
% account the time series. My targets and inputs depended on the frames at
% which they were captured which, giving it some thought, seems superfluous
% to detect phases. Rather, I should be defining the targets in such a way
% that as soon as an instrument's binary signal is detected as 1 or 0, a
% phase can be predicted -- no need to include time; just if the tool is
% present or not. If it is = phase x, if not = phase xi, etc.

% For example, if the entire input signal array (containing the 7 tool
% signals) only contains 1s from the top row, this corresponds to the
% grasper tool being present and other tools not being present. This
% strongly suggests that the preparation phase is underway, and the hope is
% that the nn will recognise this.

% Want to take tool signals from 1st surgery and see if phase1 can be
% recognised. In other words, if the network receives any tool but tool 1
% (column 1), it will give an output ~=0. When it does receive tool 1's
% signal, it classifies it as being part of phase 1.
close all;
%choice = input('Which surgery do you wish to classify?: \n');
X = toolInputs; 

% Define the targets for each of the 7 phases. Each row represents the 
% targets for each phase. Each column is the status of a particular 
% instrument during that phase: 1 means tool present in phase; 0 not
% present. 

% First attempt. Doesnt really do what I want it to do, because it takes 1
% column rather than the row which means its not testing the phase. 
%T = [1 0 0 0 0 0 0; 1 0 1 0 0 0 0; 1 0 0 1 1 0 0;...
%     1 0 1 0 0 1 0; 1 0 0 0 0 0 1; 1 1 0 0 0 1 1; 1 0 0 0 0 0 1];

% Second attempt (30/8)
% Try following example of wine dataset whereby targets should stretch for
% as long as the tool signal. That way, phases can be labelled properly.
% (might not work, but try it anyway). Not really incorporating time, but
% rather setting up targets for when tools are expected to come into play
% for each phase. 

T = phaseTargets2;

%%
% Two-layer (i.e. one-hidden-layer) feed forward neural networks can learn
% any input-output relationship given enough neurons in the hidden layer.
% Layers which are not output layers are called hidden layers.
% We will try a single hidden layer of 10 neurons for this example. In
% general, more difficult problems require more neurons, and perhaps more
% layers. Simpler problems require fewer neurons.

net = patternnet(10);

% Train the network and view performance plots
[net,tr] = train(net,X,T);
nntraintool % access app for plots
% view(net) % visualise network architecture

% Plot networks error improvement performance. Measured as MSE. 
plotperform(tr);

%% Testing the Neural Network
% The mean squared error of the trained neural network can now be measured
% with respect to the testing samples. This will give us a sense of how
% well the network will do when applied to data from the real world.
%
% The network outputs will be in the range 0 to 1, so we can use *vec2ind*
% function to get the class indices as the position of the highest element
% in each output vector.

testX = X(:,tr.testInd);
testT = T(:,tr.testInd);

testY = net(testX);
testIndices = vec2ind(testY);

%% RESULTS

% To be honest, dont really expect great results; most work that uses nn's
% get classification accuracies of around 40-50%, so anything along those
% lines would be ideal for preliminary results. See the paper below for an 
% example of a cross-validation study on different methods/algorithms that includes NNs:
%   'Automatic phase recognition in pituitary surgeries by microscope image
%   classification', 2010. Pierre Jannin et al.
%=====================================================%

%% Confusion matrix

% Another measure of how well the neural network has fit the data is the
% confusion plot.  Here the confusion matrix is plotted across all samples.
%
% The confusion matrix shows the percentages of correct and incorrect
% classifications.  Correct classifications are the green squares on the
% matrices diagonal.  Incorrect classifications form the red squares.
%
% If the network has learned to classify properly, the percentages in the
% red squares should be very small, indicating few misclassifications.
%
% If this is not the case then further training, or training a network
% with more hidden neurons, would be advisable.

plotconfusion(testT,testY)

%% Overall percentage of correct and incorrect classification

[c,cm] = confusion(testT,testY);
fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);

%% ROC Curve
% A third measure of how well the neural network has fit data is the
% receiver operating characteristic plot.  This shows how the false
% positive and true positive rates relate as the thresholding of outputs
% is varied from 0 to 1.
%
% The farther left and up the line is, the fewer false positives need to
% be accepted in order to get a high true positive rate.  The best
% classifiers will have a line going from the bottom left corner, to the
% top left corner, to the top right corner, or close to that.

plotroc(testT,testY)


