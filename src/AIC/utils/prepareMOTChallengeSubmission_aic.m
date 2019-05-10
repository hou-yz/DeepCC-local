function prepareMOTChallengeSubmission_aic(opts)
% Prepare submission file duke.txt

submission_data = [];
for scene = opts.seqs{opts.sequence}
    for iCam = opts.cams_in_scene{scene}
        filename = sprintf('%s/%s/L3-identities/cam%d_%s.txt', opts.experiment_root, opts.experiment_name, iCam, opts.folder_by_seq{opts.sequence});
        data = dlmread(filename);
        data(:,[1 2]) = data(:,[2 1]);
        data = [iCam*ones(size(data,1),1), data];
        data = data(:,[1:7]);
        submission_data = [submission_data; data];
    end
end

too_big = abs(submission_data)>10^4;
too_big = find(sum(too_big,2));
submission_data(too_big,:) = [];

% Write duke.txt
dlmwrite(sprintf('%s/%s/aic19.txt',opts.experiment_root, opts.experiment_name), int32(submission_data), 'delimiter', ' ','precision',6);
end
