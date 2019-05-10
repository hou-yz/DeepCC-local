function [st_affinity,impossibility] = aic_L3_motion_score(opts,trajectories,start_indicator,end_indicator)
%AIC_L3_MOTION_SCORE Summary of this function goes here
%   Detailed explanation goes here
st_affinity = zeros(length(trajectories));
impossibility   = zeros(length(trajectories));
iCams = [trajectories.camera];

for idx1 = 1 : length(trajectories)-1
    for idx2 = idx1+1 : length(trajectories)
        
        % temporal ordering is required here (A observed before B)
        if trajectories(idx1).data(1, 9) < trajectories(idx2).data(end, 9)
            A = trajectories(idx1);
            B = trajectories(idx2);
            former_idx = idx1; latter_idx = idx2;
        else
            A = trajectories(idx2);
            B = trajectories(idx1);
            former_idx = idx2; latter_idx = idx1;
        end
        
        if ~(start_indicator(latter_idx) && end_indicator(former_idx))
            continue;
        else
            % former:A -> end speed
            indices = max(1,size(A.data,1)-inf):size(A.data,1);
            A_speed = A.data(indices(end),7:8) - A.data(indices(1),7:8);
            A_speed = A_speed/(A.data(indices(end),9)-A.data(indices(1),9)+10^-12)*10;
            % latter:B -> start speed
            indices = 1:min(1+inf,size(B.data,1));
            B_speed = B.data(indices(end),7:8) - B.data(indices(1),7:8);
            B_speed = B_speed/(B.data(indices(end),9)-B.data(indices(1),9)+10^-12)*10;
        end
        
        [overlapping_frames,ia,ib] = intersect(A.data(:,9),B.data(:,9));
        if false%~isempty(overlapping_frames)
            xDetected = [A.data(ia,7);B.data(ib,7)];
            yDetected = [A.data(ia,8);B.data(ib,8)];
            frames    = [A.data(ia,9);B.data(ib,9)];
            
            xModel = polyfit(frames,xDetected,2);
            xPredicted = polyval(xModel, frames);
            yModel = polyfit(frames,yDetected,2);
            yPredicted = polyval(yModel, frames);
            pos_diff = sqrt((xPredicted - xDetected).^ 2 + (yPredicted - yDetected).^ 2);
            smoothnessLoss = mean(pos_diff);
            
%             figure(4)
%             clf('reset');
%             hold on
%             scatter(xDetected,yDetected)
%             scatter(xPredicted,yPredicted,'fill')
%             legend('Detected Position','Predicted Position')
%             daspect([1 1 1])

            if smoothnessLoss > 20
                impossibility(idx1,idx2) = true;
            end
        else
            AB_v_cos = pdist2(A_speed,B_speed,'cosine');
            AB_v_euc = pdist2(A_speed,B_speed,'euclidean');
            distance = B.data(end, [7 8]) - A.data(1, [7 8]);
            N_speed = distance./(B.data(end, 9) - A.data(1, 9)+10^-12)*10;
            AN_v_cos = pdist2(A_speed,N_speed,'cosine');
            BN_v_cos = pdist2(B_speed,N_speed,'cosine');

            %if sum(A_speed.^2) > opts.identities.speed_limit(2) & sum(B_speed.^2) > opts.identities.speed_limit(2) & sum(N_speed.^2) > opts.identities.speed_limit(2)
            impossibility(idx1,idx2) = AB_v_cos > pi/2 | AN_v_cos > pi/2 | BN_v_cos > pi/2;
            %end
            impossibility(idx1,idx2) = impossibility(idx1,idx2) | AB_v_euc > opts.identities.speed_limit(1);
        end
        
    end
end
impossibility = impossibility + impossibility';

impossibility(iCams == iCams')  = 1;
impossibility(logical(eye(length(trajectories))))  = 0;
impossibility = logical(impossibility);
end

