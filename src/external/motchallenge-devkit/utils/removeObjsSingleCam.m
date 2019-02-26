function resMatProc = removeObjsSingleCam(resMat)

resIDMultiCam = [];
resID1stCam = [];

% Check whether each ID passes through multiple cameras
for camInd = 1:length(resMat)
    resdata = resMat{camInd};
    for objInd = 1:size(resdata, 1)
        ID = resdata(objInd, 2);
        
        while ID > length(resIDMultiCam)
            resIDMultiCam = [resIDMultiCam, false];
        end
        
        while ID > length(resID1stCam)
            resID1stCam = [resID1stCam, -1];
        end
        
        if resID1stCam(ID) < 0
            resID1stCam(ID) = camInd;
        end
        
        if ~resIDMultiCam(ID) && resID1stCam(ID) >= 0 && resID1stCam(ID) ~= camInd
            resIDMultiCam(ID) = true;
        end
    end
end

resMatProc = [];

% Remove ID(s) that only pass through single camera
for camInd = 1:length(resMat)
    resdataProc = [];
    resdata = resMat{camInd};
    for objInd = 1:size(resdata, 1)
        ID = resdata(objInd, 2);
        
        if resIDMultiCam(ID)
            resdataProc = [resdataProc; resdata(objInd, :)];
        end
    end
    
    if ~isempty(resdataProc)
        resMatProc{camInd} = resdataProc;
    else
        resMatProc{camInd} = resdata;
    end
end

end