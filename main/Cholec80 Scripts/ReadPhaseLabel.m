function [gt] = ReadPhaseLabel(phaseFile)

fid = fopen(phaseFile,'r');

% read the header first
tline = fgets(fid); 

% read the labels
[gt] = textscan(fid,'%d\t%s');

fclose(fid);

end

