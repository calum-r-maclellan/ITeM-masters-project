% Function to create target data for NN.
% Allocates phase numbers to timesteps of phase labels (25fps)
% and converts this to the 1fps of tool signal acquisition. Doing so reduces
% the phase matrix by 25x its size and fits all the phase separations into
% the length of the tool data. Takes the previous length and assigns the 
% next phase to length + 1 all the way to the next length. 

% Clarify:
% For example, the length of the first phase is 1x525 which is /25 = 1x21 for tools.
% Program sees that jj=1 so assigns 1s from 1x21 and continues to next 
% iteration. 
% At jj=2, it takes the previous phase length (21) adds 1 (so then at frame 21+1=22) and
% assigns the next phase number (2) from here to the next length 
% (which is 673). Thus, there are a line of 2s from 22->673 to show the 2nd phase.
% It continues doing this until all the phases have been labelled.  

% With this, TARGETS can be created to identify where different tools are
% in different phases for the NN. 

function phaseTargets = PhaseTargetFunc(T_ph,numSurgeries,nPhases)

phaseTargets = [];

for ii = 1: numSurgeries 
    for jj = 1: nPhases
        if ii > 1
            A(1: T_ph(ii,jj)) = jj;
            phaseTargets = [phaseTargets A]; 
            clear A; % to make sure that phaseTargets gets updated with new phase info
        elseif jj == 1      
            phaseTargets(1: T_ph(ii,jj)) = jj;
        else
            % This bit is just for surgery 1. Is redundant for the rest.
            B(1: T_ph(ii,jj)) = jj;
            phaseTargets = [phaseTargets B];
            clear B;
        end
    end
end
 
end % function end