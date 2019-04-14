function camera_pos_mapping = get_fisheye_mapping(opts)
%GET_FISHEYE_MAPPING Summary of this function goes here
%   Detailed explanation goes here
camera_pos_mapping = cell(1,40);

format long
for iCam = [5,35]
    param = opts.projection{iCam};
    load(fullfile('src/AIC',sprintf('C%02d_Mapping.mat',iCam)))
    camera_pos_mapping{1,iCam} = struct('undistort2pix_x',x','undistort2pix_y',y');
end
end

