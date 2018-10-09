
iCam_i_reintro_time_avgs = zeros(1,8);
iCam_i_reintro_time_95ths = zeros(1,8);
iCam_i_reintro_time_995ths = zeros(1,8);
iCam_i_reintro_time_hists = cell(8,1);
iCam_i_reintro_times = cellmat(8,1,0,0,0);
for i=1:8
    for j=1:8
        if isempty(iCam_i_reintro_time_lists{i,j})
            continue
        end
        iCam_i_reintro_times{i} = [iCam_i_reintro_times{i},iCam_i_reintro_time_lists{i,j}];
    end
    iCam_i_reintro_time_avgs(i) = mean(iCam_i_reintro_times{i});
    
    iCam_i_reintro_time_95ths(i) = prctile(iCam_i_reintro_times{i},95);
    iCam_i_reintro_time_995ths(i) = prctile(iCam_i_reintro_times{i},99.5);
%     figure()
%     iCam_i_reintro_time_hists{i} = histogram(iCam_i_reintro_times{i});
end
sum_outage_icams = sum(consecutive_cam_matrix,2);
consider_icam_matrix = consecutive_cam_matrix>sum_outage_icams*0.005;
disp(consider_icam_matrix)
fprintf('%.0f,',iCam_i_reintro_time_95ths)
fprintf('%.0f,',iCam_i_reintro_time_995ths)