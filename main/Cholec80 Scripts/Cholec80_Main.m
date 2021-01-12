

%% Cholec80_Main.m

% Main script containing the functions that read in the cholec80.txt
% dataset, calculate some useful statistics for phase recognition, and 
% prepare the data for the NARX neural network.


%% Initialise
% REMEMBER TO ADD cholec80_Tool+Phase_annotations FOLDER TO MATLAB PATH!!

clear; close all;

% Specify the location of the data
myFolder  = '/Volumes/ENGD_REPO/MEng Project/ITeM2017_CalumMacLellan/Data/cholec80_Tool+Phase_annotations/tool_annotations';
myFolder2 = '/Volumes/ENGD_REPO/MEng Project/ITeM2017_CalumMacLellan/Data/cholec80_Tool+Phase_annotations/phase_annotations';

%% 'Find' and 'read' functions

% Find and specify the cholec80.txt files 
theFiles  = FindCholec80ToolFiles(myFolder);
theFiles2 = FindCholec80PhaseFiles(myFolder2);

% Set up parameters
% Give the user flexibility to choose number of datasets to evaluate
disp('The cholec80 dataset contains 80 laparoscopic cholecystectomy phase and tool annotations for 80 videos.');
numSurgeries = input('How many datasets do you wish to use?\n>> '); % ...of 80 cholec. datasets

if numSurgeries == 0
   disp('Program terminated by user.');
   return;
end

numTools = 7;      % 7 tools - open surgeryTools cell for labels
toolSignals  = cell(numSurgeries,numTools);
surgeryTools = cell(1,numTools);
nPhases  = 7; 
phaseNames = {'Preparation',          'CalotTriangleDissection', ...
              'ClippingCutting',      'GallbladderDissection', ...
              'GallbladderPackaging', 'CleaningCoagulation', ...
              'GallbladderRetraction'
             };
cholecPhases = cell(numSurgeries,2);
phaseLength  = zeros(numSurgeries,nPhases);

% Function call: read in tools and phase data
[toolSignals,surgeryTools] = ReadCholec80ToolFiles(theFiles,toolSignals,surgeryTools,numSurgeries);
[cholecPhases,phaseLength] = ReadCholec80PhaseFiles(theFiles2,cholecPhases,phaseLength,numSurgeries,phaseNames);

disp('=========================================================');
disp('Data importing process complete.');
disp('Tool and Phase data successfully imported into workspace.');
disp('=========================================================');

%% Phase data analysis: calculating the statistics
% Set up function to calculate the total surgery time and individual phase
% durations (in secs). Then take the means of all the surgery and phase times, and
% calculate standard deviations. This will give me an estimate of a
% standard surgery with a mean±std value for phases and total surgery time
% => for phase recognition later on!!


% Example: Basic surgery duration calculation. Phases were captured at 25fps, so
% divide final frame by 25. This gives total surgery time in seconds (x). Then
% divide by 60 to get the total time in minutes. 
%  x = phaseLength(1,7)/25; % 1st surgery: 1733.04 seconds
% keep in this format (s) for each phase and surgery. Then convert to mins/secs when all
% samples are calculated for tabulation.  

% Quick sanity check: convert to mins and secs (mins.secs)
%  t = x/60; % gives 28.884 minutes
%  timeMins = floor(t) + (t-floor(t))*60/100; 
% which is 28.5304, or 28mins 53.04secs => compare with timestamp.txt
% ==> Same value == calculation works. 

%% Durations function

T_ph  = zeros(numSurgeries,nPhases);            % phase durations array
phaseTimeStamps = zeros(numSurgeries,nPhases);  % phase time stamps array
T_sur = zeros(numSurgeries,1);                  % total surgery times array

% Determine the phase and surgery durations
[T_ph,T_sur,phaseTimeStamps] = DurationCalculator(T_ph,T_sur,phaseTimeStamps,phaseLength,numSurgeries);

%% Statistics function (for mean and std)
% Calculate the stats for each phase and surgery and plots phase durations. 
% Phases:
% 1: Preparation, 2: CalotTriangleDissection, 3: Clipping and Cutting,
% 4: Gallbladder Dissection, 5: Gallbladder Packaging,
% 6: Cleaning and Coagulation, 7: Gallbladder Retraction.

meanPhaseDur = zeros(1,nPhases); 
stdPhaseDur  = zeros(1,nPhases);

for i = 1: nPhases
   
    % Phases
    meanPhaseDur(i) = mean(T_ph(:,i));
    stdPhaseDur(i)  = std(T_ph(:,i));
    
    % Surgery
    meanSurgDur = mean(T_sur);
    stdSurgDur  = std(T_sur); 
   
end

%% Building a standard surgery using means and std's 

% Bar graph for phase duration statistics
figure(1);
clf;
axis([0 7.5,0 1700]);
xlabel('\bfPhase Number');
ylabel('\bfTime [s]');
hold on
bar(meanPhaseDur,'BarWidth',0.6);
errorbar(meanPhaseDur,stdPhaseDur,'x');
hold off

%% EXTRA PART: Convert from seconds to min:s
% Makes it easier to visualise phase times

meanPhaseTime = zeros(1,nPhases); 
stdPhaseTime  = zeros(1,nPhases);

for i = 1:length(meanPhaseDur)
    t = meanPhaseDur(i)/60;
    meanPhaseTime(i) = floor(t) + (t - floor(t))*60/100;
    x = stdPhaseDur(i)/60;
    stdPhaseTime(i)  = floor(x) + (x - floor(x))*0.6;
end
 
tS = meanSurgDur/60;
meanSurgTime = floor(tS) + (tS - floor(tS))*60/100; % mean = 38mins 26secs
xS = stdSurgDur/60;
stdSurgTime  = floor(xS) + (xS - floor(xS))*0.6;    % std  = 17mins 05secs

%=======================================================%
%    END OF INITIAL DATA ACQUISITION AND PROCESSING     %
%=======================================================%

%% Tool presence: identifying the tools present in each phase
% To identify the tools present in certain phases, need to count the
% number of 1s (of each tool) between the time intervals that define each
% phase: use the new cell array phaseTimeStamps for this.
clear i 
grasperCount   = zeros(numSurgeries,nPhases); meanGrasper   = zeros(1,nPhases);
bipolarCount   = zeros(numSurgeries,nPhases); meanBipolar   = zeros(1,nPhases);
hookCount      = zeros(numSurgeries,nPhases); meanHook      = zeros(1,nPhases);
scissorsCount  = zeros(numSurgeries,nPhases); meanScissors  = zeros(1,nPhases);
clipperCount   = zeros(numSurgeries,nPhases); meanClipper   = zeros(1,nPhases);
irrigatorCount = zeros(numSurgeries,nPhases); meanIrrigator = zeros(1,nPhases);
specBagCount   = zeros(numSurgeries,nPhases); meanSpecBag   = zeros(1,nPhases);

for i = 1: numSurgeries
k = 1; % initialise k as 1 so that the inner loop keeps its current value of k
       % during each jth iteration, and so that k resets for each ith
       % iteration
  
    for j = 1: nPhases 
    % Reset counters each jth iteration to prevent adding counts to next phases
    t1 = 0; t2 = 0; t3 = 0; t4 = 0; t5 = 0; t6 = 0; t7 = 0; 
    
         for k = k: phaseTimeStamps(i,j) % length of phase (s or 1fps)
                  
                  % Create separate if statements for each tool. There has
                  % to be a more elegant way to do this, but at least it
                  % works.
                  if toolSignals{i,1}(k) == 1
                     t1 = t1 + 1;
                     grasperCount(i,j) = t1;
                  end
                    
                  if toolSignals{i,2}(k) == 1
                     t2 = t2 + 1;
                     bipolarCount(i,j) = t2;
                  end           
    
                  if toolSignals{i,3}(k) == 1
                     t3 = t3 + 1;
                     hookCount(i,j) = t3;
                  end
                     
                  if toolSignals{i,4}(k) == 1
                     t4 = t4 + 1;
                     scissorsCount(i,j) = t4;
                  end
                  
                  if toolSignals{i,5}(k) == 1
                     t5 = t5 + 1;
                     clipperCount(i,j) = t5;
                  end
                  
                  if toolSignals{i,6}(k) == 1
                     t6 = t6 + 1;
                     irrigatorCount(i,j) = t6;
                  end
                  
                  if toolSignals{i,7}(k) == 1
                     t7 = t7 + 1;
                     specBagCount(i,j) = t7;
                  end
         end
    % Calculate the mean occurrence of each tool in each phase for each surgery
    meanGrasper(j)   = mean(grasperCount(:,j));
    meanBipolar(j)   = mean(bipolarCount(:,j));
    meanHook(j)      = mean(hookCount(:,j));
    meanScissors(j)  = mean(scissorsCount(:,j));
    meanClipper(j)   = mean(clipperCount(:,j));
    meanIrrigator(j) = mean(irrigatorCount(:,j));
    meanSpecBag(j)   = mean(specBagCount(:,j));
    end
end

% Now need to categorise each instrument occurrence into phases to see
% which tools are present in what phase. Do this by plotting the mean
% occurrence against phase number to see where most weight for each tool
% lies.
figure(2);
axis([1 7,0 1000]);
plot(meanGrasper,'bx-');
hold on
    plot(meanBipolar,'rx-');
    plot(meanHook,'cx-');
    plot(meanScissors,'kx-');
    plot(meanClipper,'gx-');
    plot(meanIrrigator,'yx-');
    plot(meanSpecBag,'mx-');
hold off
xlabel('\bfPhase Number');
ylabel('\bfTool occurrence count');
% title('Mean tool occurrence for each surgical phase');
title(['Mean tool occurrence for each phase over ' num2str(numSurgeries) ' surgeries']);
legend('Grasper','Bipolar','Hook','Scissors','Clipper','Irrigator','Specimen Bag',...
       'Location','best');

%=======================================================%
%               END OF STATISTICS SECTION               %
%=======================================================%

%% PHASE RECOGNITION by tool signals and time signatures

% Use PhaseTargetFunc function to create the inputs for the NARX neural
% network. 
% Formatted by 
phaseTargets = PhaseTargetFunc(T_ph,numSurgeries,nPhases);   
   
% Good. Now that we have targets, I need inputs - this is the tools. 
% Need to do a similar thing as the targets, except that each tool has to
% be pushed into one row for all surgeries. Expect 7 rows representing the
% tool signals.
% Very simple 1-line code. Heres how it works:
    % For i=1: Shove all the rows of the first column of toolSignals (which
    % is the grasper tool in x surgeries) into all the columns of
    % the first row of toolInputs. This represents the data for 1
    % instrument over all x surgeries. For i=2, all the signals from
    % x surgeries are squished together and shoved onto the 2nd row
    % of toolInputs. This represents the data for two instruments over all
    % given surgeries. Carry on till 7: the last tool.
    
% This gives me 7 rows of data that each combine the signals from different
% surgeries into 1 row per tool. 

toolInputs = zeros(numTools,length(phaseTargets));
for i = 1: numTools
        toolInputs(i,:) = cell2mat(toolSignals(:,i));
end

% Check compatibilty between input and target data (must be same length) 
if size(phaseTargets) == size(toolInputs)
   disp('=============================');
   disp('Inputs and Targets compatible.');
   disp('Ready to train neural network.');
   disp('=============================');
else
   error(['ERROR:' phaseTargets '\n Inputs and targets have different sizes.']);
end

 


