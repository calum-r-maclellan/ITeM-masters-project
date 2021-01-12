function [u_tools,y_phases,u,y] = narxdatafcn(toolInputs,phaseTargets,T_sur,nTrain)
u = cell(1,80);
y = cell(1,80);
for i = 1:80
    if i > 1
       u{i} = con2seq(toolInputs(:,sum(T_sur(1:i-1))+1: sum(T_sur(1:i-1))+T_sur(i)));
       y{i} = con2seq(phaseTargets(sum(T_sur(1:i-1))+1: sum(T_sur(1:i-1))+T_sur(i)));
    elseif i == 1
       u{i} = con2seq(toolInputs(:,1:T_sur(i)));
       y{i} = con2seq(phaseTargets(1:T_sur(i)));
    end  
end

% Concatenate all the samples together
u_tools  = catsamples(u{1:nTrain},'pad');
y_phases = catsamples(y{1:nTrain},'pad');