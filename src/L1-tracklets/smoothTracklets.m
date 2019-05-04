function smoothedTracklets = smoothTracklets(opts, tracklets, segmentStart, segmentInterval, featuresAppearance, minTrackletLength, currentInterval, iCam)
% This function smooths given tracklets by fitting a low degree polynomial 
% in their spatial location

trackletIDs          = unique(tracklets(:,2));
numTracklets         = length(trackletIDs);
smoothedTracklets    = struct([]);

for i = 1:numTracklets
    mask = tracklets(:,2)==trackletIDs(i);
    detections = tracklets(mask,:);
    
    % Reject tracklets of short length
    start = min(detections(:,1));
    finish = max(detections(:,1));
    
    
    bboxs = detections(:,3:6);
    ious = bboxOverlapRatio(bboxs,bboxs);
%     if opts.visualize
%         for j = 1:size(detections,1)
%             frame = detections(j,1);
%             bbox = bboxs(j,:);
%             fig = show_bbox(opts,iCam,frame,bbox);
%             figure(5)
%             imshow(fig)
%         end
%     end
    
    if (size(detections,1) < minTrackletLength) || (finish - start < minTrackletLength) || (sum(sum(ious>0)) == 0)
        continue;
    end

    intervalLength = finish-start + 1;
    
    datapoints = linspace(start, finish, intervalLength);
    frames     = detections(:,1);
    
    currentTracklet      = zeros(intervalLength,size(tracklets,2));
    currentTracklet(:,2) = ones(intervalLength,1) .* trackletIDs(i);
    currentTracklet(:,1) = [start : finish];
    
    % Fit left, top, right, bottom, xworld, yworld
    for k = 3:size(tracklets,2)
        det_points    = detections(:,k);
        motion_model  = polyfit(frames,det_points,2);
        newpoints     = polyval(motion_model, datapoints);
        currentTracklet(:,k) = newpoints';
    end
    %% realdata for aic
    if opts.dataset == 2
        unique_frames   = unique(frames);
        if length(unique_frames)<length(frames)
            currentTracklet = zeros(length(unique(frames)),8);
            for j = 1:length(unique_frames)
                frame = unique_frames(j);
                currentTracklet(j,:) = mean(detections(detections(:,1)==frame,:),1);
            end
        else
            currentTracklet = detections;
        end
    end
    
    
    % Compute appearance features
    meanFeature    = mean(cell2mat(featuresAppearance(mask,:)),1);
    centers          = getBoundingBoxCenters(currentTracklet(:,[3:6]));
    centerPoint      = mean(centers,1); % assumes more then one detection per tracklet
    % centerPointWorld = 1;% 
    centerPointWorld = mean(currentTracklet(:,[7,8]),1);
    
    smoothedTracklets(end+1).feature       = meanFeature; 
    smoothedTracklets(end).center          = centerPoint;
    smoothedTracklets(end).centerWorld     = centerPointWorld;
    smoothedTracklets(end).data            = currentTracklet;
    smoothedTracklets(end).features        = featuresAppearance(mask,:);
    smoothedTracklets(end).realdata        = detections;
    smoothedTracklets(end).mask            = mask;
    smoothedTracklets(end).startFrame      = start;
    smoothedTracklets(end).endFrame        = finish;
    smoothedTracklets(end).interval        = currentInterval;
    smoothedTracklets(end).segmentStart    = segmentStart;
    smoothedTracklets(end).segmentInterval = segmentInterval;
    smoothedTracklets(end).segmentEnd      = segmentStart + segmentInterval - 1;
    
    assert(~isempty(currentTracklet));
end
end



