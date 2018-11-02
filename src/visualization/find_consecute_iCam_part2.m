
iCam_i_reintro_time_avgs = zeros(1,8);
iCam_i_reintro_time_95ths = zeros(1,8);
iCam_i_reintro_time_995ths = zeros(1,8);
iCam_i_reintro_time_hists = cell(8,1);
iCam_i_reintro_times = cellmat(8,1,0,0,0);
all_reintro_times = [];
same_track_6000 = zeros(8);
same_track_12000 = zeros(8);
for i=1:8
    for j=1:8
        same_track_6000(i,j) = sum(same_track_intervals{i,j}<6000);
        same_track_12000(i,j) = sum(same_track_intervals{i,j}<12000);
        if isempty(iCam_i_reintro_time_lists{i,j})
            continue
        end
        iCam_i_reintro_times{i} = [iCam_i_reintro_times{i},iCam_i_reintro_time_lists{i,j}];
        all_reintro_times = [all_reintro_times,iCam_i_reintro_time_lists{i,j}];
    end
    iCam_i_reintro_time_avgs(i) = mean(iCam_i_reintro_times{i});
    
    iCam_i_reintro_time_95ths(i) = prctile(iCam_i_reintro_times{i},95);
    iCam_i_reintro_time_995ths(i) = prctile(iCam_i_reintro_times{i},99.5);
%     figure()
%     iCam_i_reintro_time_hists{i} = histogram(iCam_i_reintro_times{i});
end
figure()
% set(gca, 'XScale', 'log')
yyaxis left
reintro_freq = histogram(all_reintro_times,1000);
yyaxis right
cum_reintro_freq = cdfplot(all_reintro_times);


sum_outage_icams = sum(consecutive_cam_matrix,2);
compare_mat = repmat(diag(same_track_12000)*0.1,1,8);
compare_mat = max(compare_mat,compare_mat');
consider_icam_matrix = same_track_12000>compare_mat;
consider_icam_matrix = consider_icam_matrix-diag(diag(consider_icam_matrix));
disp(consider_icam_matrix)
% fprintf('%.0f,',iCam_i_reintro_time_95ths)
% fprintf('%.0f,',iCam_i_reintro_time_995ths)