function resdataProc = removeOutliersROI(resdata, cam, settype)

% Fetch data
if ~exist('res/AIC19/ROIs','dir')
    fprintf('Downloading ROI images...\n');
    url = 'https://drive.google.com/uc?export=download&id=1aT8rZ2sEdBIKNuDJz1EinsBpgvYobZHN';
    if ~exist('res','dir'), mkdir('res'); end
    if ~exist('res/AIC19','dir'), mkdir('res/AIC19'); end
    if ~exist('res/AIC19/ROIs','dir'), mkdir('res/AIC19/ROIs'); end
    filename = 'res/AIC19/ROIs.zip';
    if exist('websave','builtin')
      outfilename = websave(filename,url); % exists from MATLAB 2014b
    else
      outfilename = urlwrite(url, filename);
    end
    unzip(outfilename,'res/AIC19/ROIs/');
    delete(filename);
end

resdataProc = [];

% Read ROI image
roipath = sprintf('res/AIC19/ROIs/%s/c%03d/roi.jpg', settype, cam);
ROI = imread(roipath);
% ROI = rgb2gray(ROI);

% Remove outliers outside the ROI
for ind = 1:size(resdata, 1)
    flagOutlier = false;
    
    xmin = resdata(ind, 3) + 1;
    ymin = resdata(ind, 4) + 1;
    xmax = resdata(ind, 3) + resdata(ind, 5);
    ymax = resdata(ind, 4) + resdata(ind, 6);
%     xfoot = floor(resdata(ind, 3) + 0.5 * resdata(ind, 5));
%     yfoot = floor(resdata(ind, 4) + resdata(ind, 6));
    
    if xmin > 0 && xmin <= size(ROI, 2) && ymin > 0 && ymin <= size(ROI, 1)
        if ROI(ymin, xmin) < 255
            flagOutlier = true;
        end
    end
    
    if xmin > 0 && xmin <= size(ROI, 2) && ymax > 0 && ymax <= size(ROI, 1)
        if ROI(ymax, xmin) < 255
            flagOutlier = true;
        end
    end
    
    if xmax > 0 && xmax <= size(ROI, 2) && ymin > 0 && ymin <= size(ROI, 1)
        if ROI(ymin, xmax) < 255
            flagOutlier = true;
        end
    end
    
    if xmax > 0 && xmax <= size(ROI, 2) && ymax > 0 && ymax <= size(ROI, 1)
        if ROI(ymax, xmax) < 255
            flagOutlier = true;
        end
    end

%     if xfoot > 0 && yfoot > 0 && xfoot <= size(ROI, 2) && yfoot <= size(ROI, 1)
%         if ROI(yfoot, xfoot) < 255
%             flagOutlier = true;
%         end
%     end
    
    if ~flagOutlier
        resdataProc = [resdataProc; resdata(ind, :)];
    end
end

if isempty(resdataProc)
    resdataProc = resdata;
end

end