clc
clear
opts = get_opts_aic();

w_det = []; h_det = [];
w_gt = [];  h_gt = [];
min_gt_w = zeros(1,40); min_gt_h = zeros(1,40);
for scene = opts.seqs{opts.sequence}
    for iCam = opts.cams_in_scene{scene}
        gt                  = load(sprintf('%s/%s/S%02d/c%03d/gt/gt.txt', opts.dataset_path, opts.folder_by_scene{scene}, scene, iCam));
        det                 = load(sprintf('%s/%s/labeled/c%03d_det_ssd512_labeled.txt', opts.dataset_path, opts.folder_by_scene{scene}, iCam));
        min_gt_w(iCam) = min(gt(:,5)); min_gt_h(iCam) = min(gt(:,6));
        det(det(:,2)==-1,:) = [];
        gt                  = sortrows(gt,[2,1]); 
        det                 = sortrows(det,[2,1]);
        [C,ia]              = setdiff(gt(:,1:2),det(:,1:2),'rows');
        gt(ia,:)            = [];
        w_det = [w_det;det(:,5)]; h_det = [h_det;det(:,6)];
        w_gt  = [w_gt;gt(:,5)];   h_gt  = [h_gt;gt(:,6)];
        
    end
end

        w_model             = polyfit(w_det,w_gt,1);
        w_pred              = polyval(w_model, w_det);
        h_model             = polyfit(h_det,h_gt,1);
        h_pred              = polyval(h_model, h_det);