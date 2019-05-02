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
        CurrentScene = 1;
        train_test = {'train','test','train','train','test'};
    end
    methods
        function obj = MyVideoReader_aic(datasetPath)
            obj.DatasetPath = datasetPath;
            obj.Video = VideoReader(fullfile(sprintf('%s/%s/S%02d/c%03d', obj.DatasetPath, obj.train_test{obj.CurrentScene}, obj.CurrentScene, obj.CurrentCamera), 'vdo.avi'));
        end
        
        function img = getFrame(obj, scene, iCam, iFrame)
            % DukeMTMC frames are 1-indexed
            
            if iCam ~= obj.CurrentCamera || scene ~= obj.CurrentScene
                obj.CurrentScene = scene;
                obj.CurrentCamera = iCam;
                obj.PrevFrame = -1;
                obj.Video = VideoReader(fullfile(sprintf('%s/%s/S%02d/c%03d', obj.DatasetPath, obj.train_test{obj.CurrentScene}, obj.CurrentScene, obj.CurrentCamera), 'vdo.avi'));
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

