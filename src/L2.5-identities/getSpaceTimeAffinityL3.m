function [motionMatrix, impossibilityMatrix, indifferenceMatrix] = getSpaceTimeAffinityL3(source_traj,target_traj)

num_source = length(source_traj);
num_target = length(target_traj);

% constants


% parameters
par_indifference = 6000;
speedLimit       = 2.5;
frameRate        = 60000/1001;

%% create binary feasibility matrix based on speed and direction
feasibilityMatrix = zeros(num_source,num_target);
for idx1 = 1 : num_source
    for idx2 = 1 : num_target
        
        % temporal ordering is required here (A observed before B)
        if source_traj(idx1).data(1, 9) < target_traj(idx2).data(1, 9)
            A = source_traj(idx1);
            B = target_traj(idx2);
        else
            A = target_traj(idx2);
            B = source_traj(idx1);
        end
        if ~isempty(intersect(A.data(:,9),B.data(:,9)))
            continue;
        end
        
        % compute required number of frames
        distance    = sqrt(sum((A.data(end, [7 8]) - B.data(1, [7 8])).^2));
        frames_betw = abs(B.data(1, 9) - A.data(end, 9));
        min_number_of_required_frames = distance / speedLimit * frameRate;
        
%         % removed directional filtering
%         if frames_betw > min_number_of_required_frames 
%             feasibilityMatrix(idx1,idx2) = 1; % feasible association
%         end

        % compute directional information
        L1 = sqrt(sum((A.data(end, [7 8]) - B.data(  1, [7 8])).^2));
        L2 = sqrt(sum((A.data(end, [7 8]) - B.data(end, [7 8])).^2));
        L3 = sqrt(sum((A.data(  1, [7 8]) - B.data(  1, [7 8])).^2));
        % to do: same camera self transfer
        if frames_betw > min_number_of_required_frames && L1 < L2 && L1 < L3 && ~isequal(A.camera, B.camera)
            feasibilityMatrix(idx1,idx2) = 1; % feasible association
        end

    end
end
impossibilityMatrix = ~feasibilityMatrix;

%% motion information
motionMatrix = zeros(num_source,num_target);
for idx1 = 1 : num_source
    for idx2 = 1 : num_target
        
        % temporal ordering is required here (A observed before B)
        if source_traj(idx1).data(1, 9) < target_traj(idx2).data(1, 9)
            A = source_traj(idx1);
            B = target_traj(idx2);
        else
            A = target_traj(idx2);
            B = source_traj(idx1);
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

%% indifference matrix
indifferenceMatrix = zeros(num_source,num_target);
frame_difference = zeros(num_source,num_target);
for idx1 = 1 : num_source
    for idx2 = 1 : num_target
        
        % temporal ordering is required here (A observed before B)
        if source_traj(idx1).data(1, 9) < target_traj(idx2).data(1, 9)
            A = source_traj(idx1);
            B = target_traj(idx2);
        else
            A = target_traj(idx2);
            B = source_traj(idx1);
        end
        frame_difference(idx1, idx2) = max(0, B.data(1, 9) - A.data(end, 9)); % it could happen to be negative in overlapping camera settings
        indifferenceMatrix(idx1,idx2) = sigmf(frame_difference(idx1,idx2), [0.001 par_indifference/2]);

    end
end



