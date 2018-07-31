%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DRAW RELATIONSHIPS WITHIN EACH APPEARANCE GROUP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for kk = 1 : max(appearanceGroups)
    
    relevantTracklets = indices;
    trackletCentersViewTmp = [];
    
    for iii = 1 : length(relevantTracklets)
        data =  cell2mat({tracklets(relevantTracklets(iii)).data});
        centerIndex = 1 + ceil(size(data,1)/2);
        
        trackletCentersViewTmp = [trackletCentersViewTmp; getBoundingBoxCenters(data(centerIndex, 3 : 6))]; %#ok
    end
    
    pairsLocal = combnk(1:length(relevantTracklets),2);
    pairsWindow = combnk(relevantTracklets,2);
    
    pairs_i = pairsLocal(:,1);
    pairs_j = pairsLocal(:,2);
    
    points1 = trackletCentersViewTmp(pairs_i,:);
    points2 = trackletCentersViewTmp(pairs_j,:);
    
    myColorMap = NegativeEnhancingColormap(128, [-1 1], [0 0 1], [1 0 0], 1);
    
    if size(pairsLocal, 1) >2
        
        for p = 1 : length(pairsLocal)
            
            pts = [ points1(p,:); points2(p,:) ];
            correlation = correlationMatrix(pairsLocal(p,1),pairsLocal(p,2));
            
            linecolor = [0 1 0];
            
            if correlation == -inf || correlation == 10000 || correlation == 0
                continue;
            end
            
            if correlation == 10000 || correlation > 1
                linecolor = [0 1 0];
            end
            
            if correlation >= -10 && correlation <=10
                colorindex = int32( 1 + (1 + correlation) * 63 );
                colorindex = max(colorindex,1);
                colorindex = min(colorindex, 128);
                linecolor = myColorMap(colorindex,:);
            end
            
            line(pts(:,1),pts(:,2),'color',linecolor);
            hold on;
            
        end
        
        pause(0.1);
        
    end
    
end

hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%