classdef MyVideoReader < handle
% Example use
%
% reader = DukeVideoReader('F:/datasets/DukeMTMC/');
% camera = 2;
% frame = 360720; 
% figure, imshow(reader.getFrame(camera, frame));
    
    properties
        NumCameras = 8;
        NumFrames =  [359580, 360720, 355380, 374850, 366390, 344400, 337680, 353220];
        DatasetPath = '';
        CurrentCamera = 1;
        PrevCamera = 1;
        PrevFrame = 0;
        Video = [];
        lastFrame = [];
    end
    methods
        function obj = MyVideoReader(datasetPath)
            obj.DatasetPath = datasetPath;
            obj.Video = VideoReader(fullfile(obj.DatasetPath, 'videos', sprintf('camera%d.mp4', obj.CurrentCamera)));
        end
        
        function img = getFrame(obj, iCam, iFrame)
            % DukeMTMC frames are 1-indexed
            assert(iFrame > 0 && iFrame <= obj.NumFrames(iCam),'Frame out of range');
            
            if iCam ~= obj.CurrentCamera
                obj.CurrentCamera = iCam;
                obj.PrevFrame = -1;
                obj.Video = VideoReader(fullfile(obj.DatasetPath, 'videos', sprintf('camera%d.mp4', obj.CurrentCamera)));
            end
            
            if iFrame ~= obj.PrevFrame + 1
                obj.Video.CurrentTime = iFrame / obj.Video.FrameRate;
            end
            if abs(obj.Video.CurrentTime*obj.Video.FrameRate - iFrame) >= 2
                obj.Video.CurrentTime = iFrame / obj.Video.FrameRate;
            end
            img = readFrame(obj.Video);
            
            % Keep track of last read
            obj.PrevCamera = iCam;
            obj.PrevFrame = iFrame;
        end
        
    end
end

