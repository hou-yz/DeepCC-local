function [smoothnessMatrix] = aic_SmoothnessMatrix(trackletData, intervalLength)

smoothness = zeros(length(trackletData));

startFrame = zeros(1,length(trackletData));
endFrame = zeros(1,length(trackletData));
for i = 1:length(trackletData)
startFrame(i) = min(trackletData{i}(:,1));
endFrame(i) = max(trackletData{i}(:,1));
end

start_le_end = startFrame>=endFrame';
% start_le_end = logical(triu(ones(length(trackletData)),1));

% sigma = 8;

for i = 1:length(trackletData)
    if isempty(trackletData{i})
        continue
    end
    
    pos_x_i = trackletData{i}(:, 7);
    pos_y_i = trackletData{i}(:, 8);
    frame_i = trackletData{i}(:, 1);
%     bbox_size_i = sqrt(trackletData{i}(:, 5).^ 2 + trackletData{i}(:, 6).^ 2);

    
    for j = 1:length(trackletData)
        
        if ~start_le_end(i,j) || i==j
            continue
        end
        
        pos_x_j = trackletData{j}(:, 7);
        pos_y_j = trackletData{j}(:, 8);
        frame_j = trackletData{j}(:, 1);
%         bbox_size_j = sqrt(trackletData{j}(:, 5).^ 2 + trackletData{j}(:, 6).^ 2);

        
        
        xDetected = [pos_x_i;pos_x_j]; yDetected = [pos_y_i;pos_y_j]; 
        frames = [frame_i;frame_j];
%         bbox_size = [bbox_size_i;bbox_size_j];
        
        %% fit with gp regression
%         % all parameters are same as cvpr2018 workshop
%         xModel = fitrgp(frames, xDetected, 'Basis', 'linear',...%'ComputationMethod','v',...
%         'FitMethod', 'exact', 'PredictMethod', 'exact', 'Sigma', sigma,...
%         'ConstantSigma', true, 'KernelFunction', 'matern52', 'KernelParameters', [1000,1000]);
%         yModel = fitrgp(frames, yDetected, 'Basis', 'linear',...%'ComputationMethod','v',...
%         'FitMethod', 'exact', 'PredictMethod', 'exact', 'Sigma', sigma,...
%         'ConstantSigma', true, 'KernelFunction', 'matern52', 'KernelParameters', [1000,1000]);
%         
%         xPredicted = predict(xModel, frames);
%         yPredicted = predict(yModel, frames);
        
        %% fit with polyval
        xModel = polyfit(frames,xDetected,2);
        xPredicted = polyval(xModel, frames);
        yModel = polyfit(frames,yDetected,2);
        yPredicted = polyval(yModel, frames);
%         motion_model  = fitrgp(frames,det_points,'Basis','linear','FitMethod','exact','PredictMethod','exact');
%         newpoints     = resubPredict(motion_model);
        
%         figure(4)
%         clf('reset');
%         hold on
%         scatter(xDetected,yDetected)
%         scatter(xPredicted,yPredicted,'fill')
        
        %% diff
%         considered_frame = (frames>frame_i(end)-intervalLength) .* (frames<frame_j(1)+intervalLength);
%         considered_frame = frames(-intervalLength+length(frame_i):intervalLength+length(frame_i));
        considered_frame = frames;
%         pos_diff = sqrt(((xPredicted - xDetected).^ 2 + (yPredicted - yDetected).^ 2)./ bbox_size.^ 2);
        pos_diff = sqrt((xPredicted - xDetected).^ 2 + (yPredicted - yDetected).^ 2);
        smoothnessLoss = mean(pos_diff(logical(considered_frame)));
        smoothness(i,j) = smoothnessLoss;
    end
end
smoothness = smoothness + smoothness';

% smoothnessMatrix = 1./exp(10*smoothness)-0.5;
smoothnessMatrix = 1 - smoothness;
smoothnessMatrix(~start_le_end) = 0;
smoothnessMatrix = smoothnessMatrix + smoothnessMatrix';
smoothnessMatrix = max(smoothnessMatrix,-1);
end

