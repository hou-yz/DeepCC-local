function  [edge_weight,spatialGroup_max,det_ids,to_remove_inds] = DET_L1_processing(opts, originalGTs, detections, startFrame, endFrame, spatialGroup_max)
% CREATETRACKLETS This function creates short tracks composed of several detections.
%   In the first stage our method groups detections into space-time groups.
%   In the second stage a Binary Integer Program is solved for every space-time
%   group.

%% DIVIDE DETECTIONS IN SPATIAL GROUPS

% Find detections for the current frame interval
edge_weight = [];

det_ids=zeros(size(detections,1),1);

to_remove_inds = [];
% Skip if no more than 1 detection are present in the scene
if isempty(originalGTs) || isempty(detections)
    return; 
end


for window_frame = startFrame:endFrame
        gts_in_window_index = find(originalGTs(:,1)==window_frame);
        dets_in_window_index = find(detections(:,1)==window_frame);
        if isempty(gts_in_window_index)
            to_remove_inds = [to_remove_inds;dets_in_window_index];
            continue
        end
        gt_pids = unique(originalGTs(gts_in_window_index,2));
        bbox_gt_s = originalGTs(gts_in_window_index,3:6);
        bbox_det_s = detections(dets_in_window_index,3:6);
        IoUs=bboxOverlapRatio(bbox_gt_s,bbox_det_s);
        [m,i]=max(IoUs);
        if length(gt_pids)==1
            i=1;
        end            
        det_pid = gt_pids(i);
        det_pid(m<0.2)=-1;
        % indices to unique values
        [~, ind] = unique(det_pid);
        % duplicate indices
        duplicate_ind = setdiff(1:length(det_pid), ind);
        % duplicate values
        duplicate_value = det_pid(duplicate_ind);
        unique_duplicate_value = unique(duplicate_value);
        to_remove_inds=[];
        for j = 1:length(unique_duplicate_value)
            duplicate_pid = unique_duplicate_value(j);
            inds = find(det_pid==duplicate_pid);
            [~,to_keep_ind] = max(m(inds));
            
            inds(to_keep_ind)=[];
            to_remove_inds = [to_remove_inds;inds];
        end
        det_pid(to_remove_inds) = -1;
%         if length(det_pid)>length(unique(det_pid))
%             det_pid
%         end
        detections(dets_in_window_index,2)=det_pid;
end


detections(to_remove_inds,:)=[];
det_ids = detections(:,2);
% add bbox position jitter before extracting center & speed
bboxs = detections(:,3:6);
edge_weight = bboxOverlapRatio(bboxs,bboxs);





spatialGroupIDs = ones(size(detections,1),1);
spatialGroupIDs = spatialGroupIDs+spatialGroup_max;
spatialGroup_max = max(spatialGroupIDs);


%%
if opts.visualize
    detectionCenters = getBoundingBoxCenters(originalGTs(currentGTsIDX, 3:6));
    trackletsVisualizePart1
end
end
