function frame_offset = get_frame_offset(opts)
frame_offset = cell(1,5);
for scene = 1:5
    fpath = fullfile(opts.dataset_path,'cam_timestamp',sprintf('S%02d.txt',scene));
    fileID = fopen(fpath,'r');
    line1 = textscan(fileID, '%s%f\r\n');
    time_offsets = zeros(1,40);
    time_offsets(opts.cams_in_scene{scene}) = line1{2};
    frame_offsets = round(time_offsets * opts.fps);
    frame_offset{scene} = frame_offsets;
end
fclose('all');
end

