function prepareMOTChallengeSubmission_aic(opts)
% Prepare submission file duke.txt

submission_data = [];
for scene = opts.seqs{opts.sequence}
    for iCam = opts.cams_in_scene{scene}
        filename = sprintf('%s/%s/L3-identities/cam%d_%s.txt', opts.experiment_root, opts.experiment_name, iCam, opts.sequence_names{opts.sequence});
        data = dlmread(filename);
        data(:,[1 2]) = data(:,[2 1]);
        data = [iCam*ones(size(data,1),1), data];
        data = data(:,[1:7]);
        submission_data = [submission_data; data];
    end
end

% Write duke.txt
dlmwrite(sprintf('%s/%s/aic19.txt',opts.experiment_root, opts.experiment_name), int32(submission_data), 'delimiter', ' ','precision',6);
end
