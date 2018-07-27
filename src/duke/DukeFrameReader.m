classdef DukeFrameReader < handle
% Assumes all frames are extracted to DukeMTMC/frames/
% Example use
%
% reader = DukeFrameReader('g:/dukemtmc/');
% camera = 2;
% frame = 360720; 
% figure, imshow(reader.getFrame(camera, frame));
    
    properties
        NumCameras = 8;
        NumFrames = [359580, 360720, 355380, 374850, 366390, 344400, 337680, 353220];
        DatasetPath = '';
    end
    methods
        function obj = DukeFrameReader(datasetPath)
            obj.DatasetPath = datasetPath;
        end
        
        function img = getFrame(obj, iCam, iFrame)
            % DukeMTMC frames are 1-indexed
            assert(iFrame > 0 && iFrame <= obj.NumFrames(iCam),'Frame out of range');
            img = imread(sprintf('%sframes/camera%d/%d.jpg',obj.DatasetPath, iCam, iFrame));
        end
        
    end
end

