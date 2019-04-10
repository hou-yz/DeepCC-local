function [velocityChangeMatrix] = getVelocityChangeMatrix(featureData)

velocityChangeMatrix = zeros(length(featureData));

for i = 1:length(featureData)
    
    if length(featureData{i}) < 1
        continue
    end
    
    x1 = featureData{i}(:, 3) + 0.5 * featureData{i}(:, 5);
    y1 = featureData{i}(:, 4) + 0.5 * featureData{i}(:, 6);
    sizeBbox1 = sqrt(featureData{i}(:, 5).^ 2 + featureData{i}(:, 6).^ 2);
    
    accX1 = x1(3:end) + x1(1:end-2) - 2 * x1(2:end-1);
    accY1 = y1(3:end) + y1(1:end-2) - 2 * y1(2:end-1);
    
    acc1 = sqrt(accX1.^2 + accY1.^2)./ sizeBbox1(2:end-1);
    [~, indexAcc1] = max(acc1);
    
    maxAccX1 = x1(indexAcc1 + 1) / norm([x1(indexAcc1 + 1), y1(indexAcc1 + 1)]);
    maxAccY1 = y1(indexAcc1 + 1) / norm([x1(indexAcc1 + 1), y1(indexAcc1 + 1)]);

    for j = i+1:length(featureData)

        if length(featureData{j}) < 1
            continue
        end
        
        x2 = featureData{j}(:, 3) + 0.5 * featureData{j}(:, 5);
        y2 = featureData{j}(:, 4) + 0.5 * featureData{j}(:, 6);
        sizeBbox2 = sqrt(featureData{j}(:, 5).^ 2 + featureData{j}(:, 6).^ 2);

        accX2 = x2(3:end) + x2(1:end-2) - 2 * x2(2:end-1);
        accY2 = y2(3:end) + y2(1:end-2) - 2 * y2(2:end-1);

        acc2 = sqrt(accX2.^2 + accY2.^2)./ sizeBbox2(2:end-1);
        [~, indexAcc2] = max(acc2);
        
        maxAccX2 = x2(indexAcc2 + 1) / norm([x2(indexAcc2 + 1), y2(indexAcc2 + 1)]);
        maxAccY2 = y2(indexAcc2 + 1) / norm([x2(indexAcc2 + 1), y2(indexAcc2 + 1)]);
        
        velocityChangeLoss = sqrt((maxAccX1 - maxAccX2)^2 + (maxAccY1 - maxAccY2)^2);
        
        velocityChangeMatrix(i, j) = velocityChangeLoss;
        velocityChangeMatrix(j, i) = velocityChangeLoss;

    end
end

velocityChangeMatrix = reshape(mapminmax(velocityChangeMatrix(:)'), size(velocityChangeMatrix));

end

