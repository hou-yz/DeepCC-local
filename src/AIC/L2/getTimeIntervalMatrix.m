function [timeIntervalMatrix] = getTimeIntervalMatrix(featureData)

timeIntervalMatrix = zeros(length(featureData));

for i = 1:length(featureData)
    
    if length(featureData{i}) < 1
        continue
    end
    
    for j = i+1:length(featureData)
        
        if length(featureData{j}) < 1
            continue
        end
        
        timeIntervalLoss = length(find(ismember(featureData{i}(:, 1), featureData{j}(:, 1)))) / 10^6;
        
        timeIntervalMatrix(i, j) = timeIntervalLoss;
        timeIntervalMatrix(j, i) = timeIntervalLoss;
        
    end
end


end

