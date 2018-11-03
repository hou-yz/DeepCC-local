function prepareMOTChallengeSubmission(opts)
% Prepare submission file duke.txt

submission_data = [];
for sequence = 3:4
    
    for iCam = 1:8
        filename = sprintf('%s/%s/L3-identities/cam%d_%s.txt', opts.experiment_root, opts.experiment_name, iCam, opts.sequence_names{sequence});
        data = dlmread(filename);
        data(:,[1 2]) = data(:,[2 1]);
        data = [iCam*ones(size(data,1),1), data];
        data = data(:,[1:7]);
        submission_data = [submission_data; data];
    end
end

% Sanity check
frameIdPairs = submission_data(:,1:3);
[u,I,~] = unique(frameIdPairs, 'rows', 'first');
hasDuplicates = size(u,1) < size(frameIdPairs,1);
if hasDuplicates
    ixDupRows = setdiff(1:size(frameIdPairs,1), I);
    dupFrameIdExample = frameIdPairs(ixDupRows(1),:);
    rows = find(ismember(frameIdPairs, dupFrameIdExample, 'rows'));

    errorMessage = sprintf('Invalid submission: Found duplicate ID/Frame pairs in result.\nInstance:\n');
    errorMessage = [errorMessage, sprintf('%10.2f', submission_data(rows(1),:)), sprintf('\n')];
    errorMessage = [errorMessage, sprintf('%10.2f', submission_data(rows(2),:)), sprintf('\n')];
    assert(~hasDuplicates, errorMessage);
end

% Write duke.txt
dlmwrite(sprintf('%s/%s/duke.txt',opts.experiment_root, opts.experiment_name), int32(submission_data), 'delimiter', ' ','precision',6);

