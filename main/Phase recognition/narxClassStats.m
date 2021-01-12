function [percentClass,phaseCount,phaseDiff,timeDelay] = narxClassStats(t2,yp1f,phaseTimes,predLength,timeDelay)

phaseCount = zeros(1,7);
phaseDiff  = zeros(1,7);
percentClass = zeros(1,7);
for i= 1:7 % run through each ith phase and count matches 
    counter=0;
    for j = 1: predLength
        if yp1f(j) == i && t2(j) == i
           counter = counter+1;
           phaseCount(i) = counter;
           phaseDiff(i)  = phaseTimes(i)-phaseCount(i); % no of mismatches
%            if i > 1
%               if yp1f(j) == i+1 
%                  switchTime = j;
%                  timeDelay(i) = abs((phaseTimes(i-1)+phaseTimes(i))-switchTime);          
%               end
%            end
        end
    end
    percentClass(i) = phaseCount(i)/(phaseCount(i)+phaseDiff(i));
end

end % end of function