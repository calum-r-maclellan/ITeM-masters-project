% Function to read in the cholec80 phase files and store the length of each
% phase for each surgery into a matrix

function [cholecPhases,phaseLength] = ReadCholec80PhaseFiles(theFiles2,cholecPhases,phaseLength,numSurgeries,phaseNames)

for ii = 1 : numSurgeries
    clear gtLabel;
    phaseFile = theFiles2{ii};
    fprintf(1,'Now reading %s\n',phaseFile);
    [gt] = ReadPhaseLabel(phaseFile);
    cholecPhases(ii,:) = gt;
    gtLabel = [];
    for jj = 1:length(phaseNames)
        gtLabel(strcmp(phaseNames{jj}, gt{2})) = jj;
        phaseLength(ii,jj) = length(gtLabel);
    end
        if ii == 80
           disp('---------------------------------');
           disp('Finished importing all phase data.');
           disp('---------------------------------');
        end 
end

end % END OF FUNCTION