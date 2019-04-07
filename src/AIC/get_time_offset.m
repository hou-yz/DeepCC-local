function time_offset = get_time_offset(opts)
time_offset = cell(1,5);
for scene = 1:5
    fpath = fullfile(opts.dataset_path,'cam_timestamp',sprintf('s%02d.txt',scene));
    fileID = fopen(fpath,'r');
    line1 = textscan(fileID, '%s%f\r\n');
    time_offset_by_icam = zeros(1,40);
    time_offset_by_icam(opts.cams_in_scene{scene}) = line1{2};
    time_offset{scene} = time_offset_by_icam;
end
end

