% Checks for a valid installation of the Gurobi optimizer.
% Without Gurobi you can still run the tracker with the AL-ICM or KL optimizer
try
    % Change this path accordingly
    run(fullfile(opts.gurobi_path, 'gurobi_setup.m'));
catch
    fprintf('\nWARNING!\n\nGurobi optimizer not found.\nYou can still run the tracker with AL-ICM or KL optimization.\n');
end