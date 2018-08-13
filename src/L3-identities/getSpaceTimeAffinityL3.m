function [motionMatrix, impossibilityMatrix, indifferenceMatrix] = getSpaceTimeAffinityMC(trajectories)

numTrajectories = length(trajectories);

% constants


% parameters
par_indifference = 6000;
speedLimit       = 2.5;
frameRate        = 60000/1001;

%% create binary feasibility matrix based on speed and direction
feasibilityMatrix = zeros(numTrajectories);
for idx1 = 1 : numTrajectories-1
    for idx2 = idx1+1 : numTrajectories
        
        % temporal ordering is required here (A observed before B)
        if trajectories(idx1).data(1, 9) < trajectories(idx2).data(1, 9)
            A = trajectories(idx1);
            B = trajectories(idx2);
        else
            A = trajectories(idx2);
            B = trajectories(idx1);
        end
        if ~isempty(intersect(A.data(:,9),B.data(:,9)))
            continue;
        end
        
        % compute required number of frames
        distance    = sqrt(sum((A.data(end, [7 8]) - B.data(1, [7 8])).^2));
        frames_betw = abs(B.data(1, 9) - A.data(end, 9));
        min_number_of_required_frames = distance / speedLimit * frameRate;
        
        % compute directional information
        L1 = sqrt(sum((A.data(end, [7 8]) - B.data(  1, [7 8])).^2));
        L2 = sqrt(sum((A.data(end, [7 8]) - B.data(end, [7 8])).^2));
        L3 = sqrt(sum((A.data(  1, [7 8]) - B.data(  1, [7 8])).^2));
        
        if frames_betw > min_number_of_required_frames && L1 < L2 && L1 < L3 && ~isequal(A.camera, B.camera)
            feasibilityMatrix(idx1,idx2) = 1; % feasible association
        end
        
    end
end
feasibilityMatrix = feasibilityMatrix + feasibilityMatrix';
impossibilityMatrix = ~feasibilityMatrix;

%% motion information
motionMatrix = zeros(numTrajectories);
for idx1 = 1 : numTrajectories-1
    for idx2 = idx1+1 : numTrajectories
        
        % temporal ordering is required here
        if trajectories(idx1).data(1, 9) < trajectories(idx2).data(1, 9)
            A = trajectories(idx1);
            B = trajectories(idx2);
        else
            A = trajectories(idx2);
            B = trajectories(idx1);
        end
        frame_difference = abs(B.data(1, 9) - A.data(end, 9)); % it could happen to be negative in overlapping camera settings
        space_difference = sqrt(sum((B.data(1, [7 8]) - A.data(end, [7 8])).^2, 2));
        needed_speed     = space_difference / frame_difference; %mpf
        speedA = sqrt(sum(diff(A.data(:, [7 8])).^2, 2)); 
        speedA = mean(speedA(max([2, end-9]):end));
        speedB = sqrt(sum(diff(B.data(:, [7 8])).^2, 2)); 
        speedB = mean(speedB(1:min([numel(speedB)-1, 10])));
        motionMatrix(idx1, idx2) = sigmf(min([abs(speedA-needed_speed), abs(speedB-needed_speed)]), [5 0])-0.53;
        
    end
end
motionMatrix = motionMatrix + motionMatrix';

%% indifference matrix
indifferenceMatrix = zeros(numTrajectories);
frame_difference = zeros(numTrajectories);
for idx1 = 1 : numTrajectories-1
    for idx2 = idx1+1 : numTrajectories
        
            % temporal ordering is required here
        if trajectories(idx1).data(1, 9) < trajectories(idx2).data(1, 9)
            A = trajectories(idx1);
            B = trajectories(idx2);
        else
            A = trajectories(idx2);
            B = trajectories(idx1);
        end
        frame_difference(idx1, idx2) = max(0, B.data(1, 9) - A.data(end, 9)); % it could happen to be negative in overlapping camera settings
        indifferenceMatrix(idx1,idx2) = sigmf(frame_difference(idx1,idx2), [0.001 par_indifference/2]);

    end
end
indifferenceMatrix = indifferenceMatrix + indifferenceMatrix';



