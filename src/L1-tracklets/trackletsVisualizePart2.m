%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SHOW SPATIAL GROUPING AND CORRELATIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rows = find(spatialGroupIDs == spatialGroupID);

figure(2);

% draw bounding box
minx = min(detectionCentersImage(rows,1));
maxx = max(detectionCentersImage(rows,1));
miny = min(detectionCentersImage(rows,2));
maxy = max(detectionCentersImage(rows,2));

x = minx - 40;
y = miny - 40;
w = maxx - minx + 80;
h = maxy - miny + 80;

rectangle('Position',[x,y,w,h],'EdgeColor','green','LineWidth',2);

labels = originalDetections(spatialGroupObservations,1);

for kk = min(labels):min(labels)
    
    rrows = rows;
    
    if isempty(rrows)
        continue;
    end
    
    pairs = combnk([1:length(rrows)],2);
    
    pairs_i = pairs(:,1);
    pairs_j = pairs(:,2);
    
    points1 = detectionCentersImage(rrows(pairs_i),:);
    points2 = detectionCentersImage(rrows(pairs_j),:);
    
    points1top = detectionCenters(rrows(pairs_i),:);
    points2top = detectionCenters(rrows(pairs_j),:);
    
    myColorMap = NegativeEnhancingColormap(128, [-1 1], ...
        [0 0 1], [1 0 0], 1);
    
    if size(pairs,1) >2
        for p = 1 : length(pairs)
            
            pts = [ points1(p,:); points2(p,:) ];
            ptstop = [ points1top(p,:); points2top(p,:) ];
            correlation = correlationMatrix(pairs_i(p),pairs_j(p));
            colorindex = int32( 1 + (1 + correlation) * 63 );
            
            
            if correlation<-1
                linecolor = [0 0 0];
            else
                linecolor = myColorMap(colorindex,:);
            end
            
            line(pts(:,1),pts(:,2),'color',linecolor);
            hold on;
        end
    end
    
    pause(0.5);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%