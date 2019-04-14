classdef MyVideoReader_aic < handle
% Example use
%
% reader = DukeVideoReader('F:/datasets/DukeMTMC/');
% camera = 2;
% frame = 360720; 
% figure, imshow(reader.getFrame(camera, frame));
    
    properties
        DatasetPath = '';
        CurrentCamera = 1;
        PrevCamera = -1;
        PrevFrame = 0;
        Video = [];
        subset_num = [1, 1, 1, 1, 1, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4];
    end
    methods
        function obj = MyVideoReader_aic(datasetPath)
            obj.DatasetPath = datasetPath;
            obj.Video = VideoReader(fullfile(sprintf('%s/train/S%02d/c%03d', obj.DatasetPath, obj.subset_num(obj.CurrentCamera), obj.CurrentCamera), 'vdo.avi'));
        end
        
        function img = getFrame(obj, iCam, iFrame)
            % DukeMTMC frames are 1-indexed
            
            if iCam ~= obj.CurrentCamera
                obj.CurrentCamera = iCam;
                obj.PrevFrame = -1;
                obj.Video = VideoReader(fullfile(sprintf('%s/train/S%02d/c%03d', obj.DatasetPath, obj.subset_num(obj.CurrentCamera), obj.CurrentCamera), 'vdo.avi'));
            end
            
            if iFrame ~= obj.PrevFrame + 1
                obj.Video.CurrentTime = iFrame / obj.Video.FrameRate;
            end
            
            if abs(obj.Video.CurrentTime*obj.Video.FrameRate - iFrame) >= 2
                obj.Video.CurrentTime = iFrame / obj.Video.FrameRate;
            end
            
            img = readFrame(obj.Video);
            
            % Keep track of last read
            obj.PrevFrame = iFrame;
            obj.PrevCamera = iCam;
        end
        
    end
end

