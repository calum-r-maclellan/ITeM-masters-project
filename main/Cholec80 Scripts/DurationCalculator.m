function [T_ph,T_sur,phaseTimeStamps] = DurationCalculator(T_ph,T_sur,phaseTimeStamps,phaseLength,numSurgeries)

% Function to take each time stamp along the video, and calculate the
% corresponding phase and total surgery time by subtracting the
% previous timestamp (j-1) with the current timestamp (j) for each phase of
% each surgery (i). 

for i = 1:numSurgeries               % 1-80 surgeries
    for j = 1:7                      % 1-7 phases
        x = phaseLength(i,j)/25;     % 25fps video acquisition. Get into seconds.
        T_ph(i,j) = round(x);        % store initial time stamp as first phase time
        phaseTimeStamps(i,j) = round(x); % round to eliminate .004 
        if j == 1                    
           continue;                 % bypass calculation for first phase since previous phase N/A
        else
        y = round(phaseTimeStamps(i,j-1));  % previous timestamp
        tDiff = round(x - y);        % calculate time difference
        T_ph(i,j) = tDiff;           % tDiff == phase duration
        end
        T_sur(i) = sum(T_ph(i,1:7)); % sum all the phase durations for each surgery to get total operation time
    end
end

end