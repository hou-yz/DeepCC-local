
colors = distinguishable_colors(1000);
% colors(1,:) = [237,177,32]/255;%yellow[126,47,142]/255;%purple
colors([2,4,5],:) = colors([3,5,4],:);
colors(3,:) = [217,83,25]/255;%orange colors(3,:);% bright green
% colors(3,:) = [126,47,142]/255;%purple[237,177,32]/255;%yellow
% colors(4,:) = colors(5,:);% bright pink
% colors(5,:) = [0,114,189]/255;%blue
% colors(6,:) = [217,83,25]/255;%orange
% colors(9,:) = [116,170,43]/255;%green
% [2,3,4,5]
% 1:length(uni_labels)
hold on
for i = 1:length(uni_labels)
    label = uni_labels(i);
    target_yd = yd(in_window_pids==label,:);
    scatter(target_yd(:, 1), target_yd(:, 2),36,colors(i,:),'filled');
%     if num_traj_per_pid(i)>1
%         scatter(target_yd(:, 1), target_yd(:, 2),60,colors(i,:));
%     else
%         scatter(target_yd(:, 1), target_yd(:, 2),36,colors(i,:),'filled');
%     end
    
    if opts.visualize
    target_tracklets = in_window_tracklet(in_window_pids==label);
    for j = length(target_tracklets):-1:1
        tracklet = target_tracklets(j);
        iCam = tracklet.iCam;
        frame = tracklet.data(round(end/2),1);
        bbox = tracklet.data(round(end/2),3:6);
        fig=show_bbox(opts,iCam,frame,bbox);
        fig = fig(end:-1:1,:,:);           %# vertical flip
        fig = imresize(fig,[map_size*2,map_size,]/3);
        fig = addborder(fig,3,colors(i,:)*255,'outer');
        
        image(target_yd(j,1),target_yd(j,2),fig)
    end
    end
end
hold off
