function [toolSignals,surgeryTools] = ReadCholec80ToolFiles(theFiles,toolSignals,surgeryTools,numSurgeries)

for ii = 1 : numSurgeries
    toolFile = theFiles{ii};
    [gt,surgeryTools] = ReadToolAnnotationFile(toolFile);
    toolSignals(ii,:) = gt;
    fprintf(1,'Now reading %s\n',toolFile);
        if ii == 80
           disp('--------------------------------');
           disp('Finished importing all tool data.');
           disp('--------------------------------');
        end 
end

end % END OF FUNCTION