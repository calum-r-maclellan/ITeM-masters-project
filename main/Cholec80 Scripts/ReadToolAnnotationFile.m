function [gt,surgeryTools] = ReadToolAnnotationFile(toolFile)

fid_gt = fopen(toolFile, 'r'); 
tline = fgets(fid_gt); 
tline = tline(1:end-1); 
surgeryTools = strsplit(tline, '\t');
surgeryTools(1) = []; % only need tool names; therefore remove 'frame' title from cell

% read the labels
gt = textscan(fid_gt, '%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d');
gt(:,1) = [];

fclose(fid_gt); % to help matlab out, close the current ii file. prevents crashing

end % END OF FUNCTION

