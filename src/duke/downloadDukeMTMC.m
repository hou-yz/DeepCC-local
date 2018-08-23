dataset = [];
dataset.numCameras = 8;
dataset.videoParts = [9, 9, 9, 9, 9, 8, 8, 9];

% Set these accordingly
dataset.savePath = 'F:/DukeMTMC/'; % Where to store DukeMTMC (160 GB)

GET_ALL               = false; % Set this to true if you want to download everything
GET_GROUND_TRUTH      = true;
GET_CALIBRATION       = true;
GET_VIDEOS            = true;
GET_DPM               = false;
GET_OPENPOSE          = true;
GET_FGMASKS           = false;
GET_REID              = true;
GET_VIDEO_REID        = false;


options = weboptions('Timeout', 60);


%% Create folder structure
fprintf('Creating folder structure...\n');
mkdir(dataset.savePath);
folders = {'ground_truth','calibration','detections','frames','masks','videos', 'detections/DPM', 'detections/openpose'};
for k = 1:length(folders)
    mkdir([dataset.savePath, folders{k}]);
end

for k = 1:dataset.numCameras
    mkdir([dataset.savePath 'frames/camera' num2str(k)]);
    mkdir([dataset.savePath 'masks/camera' num2str(k)]);
    mkdir([dataset.savePath 'videos/camera' num2str(k)]);
end

%% Download ground truth
if GET_ALL || GET_GROUND_TRUTH
    fprintf('Downloading ground truth...\n');
    filenames = {'trainval.mat', 'trainvalRaw.mat'};
    urls = {'http://vision.cs.duke.edu/DukeMTMC/data/ground_truth/trainval.mat', ...
        'http://vision.cs.duke.edu/DukeMTMC/data/ground_truth/trainvalRaw.mat'};
    for k = 1:length(urls)
        filename = sprintf('%sground_truth/%s',dataset.savePath,filenames{k});
        fprintf([filename '\n']);
        websave(filename,urls{k},options);
    end
end
%% Download calibration
if GET_ALL || GET_CALIBRATION
    fprintf('Downloading calibration...\n');
    urls = {'http://vision.cs.duke.edu/DukeMTMC/data/calibration/calibration.txt', ...
        'http://vision.cs.duke.edu/DukeMTMC/data/calibration/camera_position.txt', ...
        'http://vision.cs.duke.edu/DukeMTMC/data/calibration/ROIs.txt'};
    filenames = {'calibration.txt', 'camera_position.txt', 'ROIs.txt'};
    
    for k = 1:length(urls)
        filename = sprintf('%scalibration/%s',dataset.savePath,filenames{k});
        fprintf([filename '\n']);
        websave(filename,urls{k},options);
    end
end

%% Download OpenPose detections
if GET_ALL || GET_OPENPOSE
    for cam = 1:dataset.numCameras
        url = sprintf('http://vision.cs.duke.edu/DukeMTMC/data/detections/openpose/camera%d.mat',cam);
        filename = sprintf('%sdetections/openpose/camera%d.mat',dataset.savePath,cam);
        fprintf([filename '\n']);
        websave(filename,url,options);
    end
end

%% Download videos
if GET_ALL || GET_VIDEOS
    fprintf('Downloading videos (146 GB)...\n');
    for cam = 1:dataset.numCameras
        for part = 0:dataset.videoParts(cam)
            url = sprintf('http://vision.cs.duke.edu/DukeMTMC/data/videos/camera%d/%05d.MTS',cam,part);
            filename = sprintf('%svideos/camera%d/%05d.MTS',dataset.savePath,cam,part);
            fprintf([filename '\n']);
            websave(filename,url,options);
        end
    end
    fprintf('Data download complete.\n');
end


%% Download DPM detections
if GET_ALL || GET_DPM
    fprintf('Downloading detections...\n');
    for cam = 1:dataset.numCameras
        url = sprintf('http://vision.cs.duke.edu/DukeMTMC/data/detections/DPM/camera%d.mat',cam);
        filename = sprintf('%sdetections/DPM/camera%d.mat',dataset.savePath,cam);
        fprintf([filename '\n']);
        websave(filename,url,options);
    end
end

%% Download background masks
if GET_ALL || GET_FGMASKS
    fprintf('Downloading masks...\n');
    for cam = 1:dataset.numCameras
        url = sprintf('http://vision.cs.duke.edu/DukeMTMC/data/masks/camera%d.tar.gz',cam);
        filename = sprintf('%smasks/camera%d.tar.gz',dataset.savePath,cam);
        fprintf([filename '\n']);
        websave(filename,url,options);
    end
    
    % Extract masks
    fprintf('Extracting masks...\n');
    for cam = 1:dataset.numCameras
        filename = sprintf('%smasks/camera%d.tar.gz',dataset.savePath,cam);
        fprintf([filename '\n']);
        untar(filename, [dataset.savePath 'masks']);
    end
    
    % Delete temporary files
    fprintf('Deleting temporary files...\n');
    for cam = 1:dataset.numCameras
        filename = sprintf('%smasks/camera%d.tar.gz',dataset.savePath,cam);
        fprintf([filename '\n']);
        delete(filename);
    end
end

%% Download DukeMTMC-reID
if GET_ALL || GET_REID
    url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-reID.zip';
    filename = fullfile(dataset.savePath,'DukeMTMC-reID.zip');
    fprintf('DukeMTMC-reID.zip\n');
    websave(filename,url,options);
    unzip(filename, dataset.savePath);
end

%% Download DukeMTMC-VideoReID
if GET_ALL || GET_VIDEO_REID
    url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-VideoReID.zip';
    filename = fullfile(dataset.savePath,'DukeMTMC-VideoReID.zip');
    fprintf('DukeMTMC-VideoReID.zip\n');
    websave(filename,url,options);
    unzip(filename, dataset.savePath);
end
