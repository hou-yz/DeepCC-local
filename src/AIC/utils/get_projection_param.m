function [projection,camera_pos_mapping] = get_projection_param(opts)
projection = cell(1,40);
for iCam = 1:40
    param = struct('homography',[],'error',[],'intrinsic',[],'distortion',[]);
    fpath = fullfile(opts.dataset_path,'calibration',sprintf('c%03d',iCam),'calibration.txt');
    fileID = fopen(fpath,'r');
    tline = fgetl(fileID);
    while tline ~= -1
        if contains(tline,'Homography matrix')
            tline = erase(tline,'Homography matrix: ');
            format long
            param.homography = str2num(tline);
        elseif contains(tline,'Reprojection error')
            tline = erase(tline,'Reprojection error: ');
            format long
            param.error = str2num(tline);
        elseif contains(tline,'Intrinsic parameter matrix')
            tline = erase(tline,'Intrinsic parameter matrix: ');
            format long
            param.intrinsic = str2num(tline);
        elseif contains(tline,'Distortion coefficients')
            tline = erase(tline,'Distortion coefficients: ');
            format long
            param.distortion = str2num(tline);
        end
        tline = fgetl(fileID);
    end
    projection{iCam} = param;    
end

fclose('all');


camera_pos_mapping = cell(1,40);

format long
for iCam = [5,35]
    load(fullfile('src/AIC/utils',sprintf('C%02d_Mapping.mat',iCam)))
    camera_pos_mapping{1,iCam} = struct('undistort2pix_x',x','undistort2pix_y',y');
end
end

