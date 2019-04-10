function [smoothnessMatrix] = getSmoothnessMatrix(featureData, intervalLength)

smoothnessMatrix = zeros(length(featureData));

sigma = 8;

for i = 1:length(featureData)
    
    if length(featureData{i}) < 1
        continue
    end
    
    % loop for the sum of connection region?
    % maybe change to matrix calculation later?
    xFitModel = featureData{i}(:, 3) + 0.5 * featureData{i}(:, 5);
    % Since the coordinate system here represent relative position change,
    % I did not convert to normal system.
    yFitModel = featureData{i}(:, 4) + 0.5 * featureData{i}(:, 6);
    frameIndex = featureData{i}(:, 1);

    % all parameters are same as last year
    % using all [x, y] or [x, y] without overlap to fit model? 
    xModel = fitrgp(frameIndex, xFitModel, 'Basis', 'linear',...
        'FitMethod', 'exact', 'PredictMethod', 'exact', 'Sigma', sigma,...
        'ConstantSigma', true, 'KernelFunction', 'matern52', 'KernelParameters', [1000,1000]);
    yModel = fitrgp(frameIndex, yFitModel, 'Basis', 'linear',...
        'FitMethod', 'exact', 'PredictMethod', 'exact', 'Sigma', sigma,...
        'ConstantSigma', true, 'KernelFunction', 'matern52', 'KernelParameters', [1000,1000]);

    % I thought there is no need to find the available tracklet before
    % or after since there is a small window size in L2.
%     for j = 1:length(featureData)

%     this might be a little bit higher
    for j = i+1:length(featureData)
        if j == i
            continue
        end

        if length(featureData{j}) < 1
            continue
        end

        framesBefore = find(featureData{j}(:, 1) < featureData{i}(1, 1));
        if length(framesBefore) > intervalLength
            framesBefore = framesBefore(1: intervalLength);
        end

        framesMiddle = find(featureData{j}(:, 1) > featureData{i}(1, 1) & featureData{j}(:, 1) < featureData{i}(end, 1));

        framesAfter = find(featureData{j}(:, 1) > featureData{i}(end, 1));
        if length(framesAfter) > intervalLength
            framesAfter = framesAfter(end - intervalLength + 1 : end);
        end

        framesPredict = [framesBefore; framesMiddle; framesAfter];

        xPredicted = predict(xModel, framesPredict);
        yPredicted = predict(yModel, framesPredict);

        xDetected = featureData{j}(framesPredict, 3) + 0.5 * featureData{j}(framesPredict, 5);
        yDetected = featureData{j}(framesPredict, 4) + 0.5 * featureData{j}(framesPredict, 6);
        
        sizeDetected = sqrt(featureData{j}(framesPredict, 5).^ 2 + featureData{j}(framesPredict, 6).^ 2);

        smoothnessLoss = mean(sqrt(((xPredicted - xDetected).^ 2 + (yPredicted - yDetected).^ 2)./ sizeDetected.^ 2));

        smoothnessMatrix(i, j) = smoothnessLoss;
        smoothnessMatrix(j, i) = smoothnessLoss;

    end
end

smoothnessMatrix = reshape(mapminmax(smoothnessMatrix(:)'), size(smoothnessMatrix));

end

