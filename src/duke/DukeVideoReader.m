classdef DukeVideoReader < handle
% Example use
%
% reader = DukeVideoReader('F:/datasets/DukeMTMC/');
% camera = 2;
% frame = 360720; 
% figure, imshow(reader.getFrame(camera, frame));
    
    properties
        NumCameras = 8;
        NumFrames =  [359580, 360720, 355380, 374850, 366390, 344400, 337680, 353220];
        PartFrames = [38370,38370,38400,38670,38370,38400,38790,38370; ...
                      38370,38370,38370,38670,38370,38370,38640,38370;...
                      38370,38370,38370,38670,38370,38370,38460,38370;...
                      38370,38370,38370,38670,38370,38370,38610,38370;...
                      38370,38370,38370,38670,38370,38400,38760,38370;...
                      38370,38370,38370,38700,38370,38400,38760,38370;...
                      38370,38370,38370,38670,38370,38370,38790,38370;...
                      38370,38370,38370,38670,38370,38370,38490,38370;...
                      38370,38370,38370,38670,38370,37350,28380,38370;...
                      14250,15390,10020,26790,21060,    0,    0, 7890];
        MaxPart = [9, 9, 9, 9, 9, 8, 8, 9];
        DatasetPath = '';
        CurrentCamera = 1;
        CurrentPart = 0;
        PrevCamera = 1;
        PrevFrame = 0;
        PrevPart = 0;
        Video = [];
        lastFrame = [];
    end
    methods
        function obj = DukeVideoReader(datasetPath)
            obj.DatasetPath = datasetPath;
            obj.Video = cv.VideoCapture(sprintf('%svideos/camera%d/%05d.MTS',obj.DatasetPath,obj.CurrentCamera, obj.CurrentPart), 'API','FFMPEG');
        end
        
        function img = getFrame(obj, iCam, iFrame)
            % DukeMTMC frames are 1-indexed
            assert(iFrame > 0 && iFrame <= obj.NumFrames(iCam),'Frame out of range');
            
            ksum = 0;
            for k = 0:obj.MaxPart(iCam)
               ksumprev = ksum;
               ksum = ksum + obj.PartFrames(k+1,iCam);
               if iFrame <= ksum
                  currentFrame = iFrame - 1 - ksumprev;
                  iPart = k;
                  break;
               end
            end
            
            if iPart ~= obj.CurrentPart || iCam ~= obj.CurrentCamera
                obj.CurrentCamera = iCam;
                obj.CurrentPart = iPart;
                obj.PrevFrame = -1;
                obj.Video = cv.VideoCapture(sprintf('%svideos/camera%d/%05d.MTS',obj.DatasetPath,obj.CurrentCamera, obj.CurrentPart), 'API','FFMPEG');
            end
            
            if currentFrame ~= obj.PrevFrame + 1
                obj.Video.PosFrames = currentFrame;
                
                if obj.Video.PosFrames ~= currentFrame
                    back_frame = max(currentFrame - 31, 0); % Keyframes every 30 frames
                    obj.Video.PosFrames =  back_frame;
                    while obj.Video.PosFrames < currentFrame
                        obj.Video.read;
                        back_frame = back_frame + 1;
                    end
                end

            end
            assert(obj.Video.PosFrames == currentFrame)
            img = obj.Video.read;
            
            % Keep track of last read
            obj.PrevCamera = iCam;
            obj.PrevFrame = currentFrame;
            obj.PrevPart = iPart;
        end
        
    end
end

