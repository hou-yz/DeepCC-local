function tracklets = fft_tracklet_feat(opts, tracklets)
for ind = 1:length(tracklets)
        % FFT features
        tracked_frames = tracklets(ind).realdata(:,1);
        all_frames = tracklets(ind).data(:,1);
        feats = cell2mat(tracklets(ind).features);
        % duplicate indices
        [missed_frames,insert_pos] = setdiff(all_frames,tracked_frames);
        for j = 1:length(insert_pos)
            pos = insert_pos(j);
            feats = [feats(1:pos-1,:);zeros(1,256);feats(pos:end,:)];
        end
        fft_feats = abs(fft(feats,opts.tracklets.window_width,1))/length(tracked_frames);
        fft_feats = reshape(fft_feats(1:4,:),1,1024);
        % Add to tracklet list
        if opts.fft
            tracklets(ind).feature      = fft_feats; 
        end
end

end

