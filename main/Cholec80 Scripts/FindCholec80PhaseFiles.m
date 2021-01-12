function theFiles2 = FindCholec80PhaseFiles(myFolder2)

% Start by checking whether the pathway exists or not
if ~isdir(myFolder2)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder2);
  uiwait(warndlg(errorMessage));
  return;
end

% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder2, '*.txt');
files2 = dir(filePattern);

%% 
% For some unfathomable reason, there are a bunch of bugged files that have
% the extension '._' before the file name. Need to remove these files as
% they are completely pointless.

theFiles2 = struct2cell(files2)'; % convert to cell array to allow individual array analysis
pattern = '._'; % recognise files with '._' extensions
m = 1;
n = length(theFiles2);
while m <= n
   if contains(theFiles2{m,1},pattern)
      theFiles2(m,:) = []; % remove files with '._' pattern
      m = 1;
   else
      m = m + 1;
   end
   % create if to leave loop once all bugged files are removed
   if m == 80
      return;
   end
end

end % END OF FUNCTION