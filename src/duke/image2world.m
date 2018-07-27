function [world, imagePoints, worldPoints] = image2world( points, camera )
% Points - image point coordinates, normalized to [0,1]x[0,1]
%points(:,1) = points(:,1) * 1920;
%points(:,2) = points(:,2) * 1080;

% points(:,[2,1]) = points(:,[1,2]) ;

% addpath('../../mexopencv/')
% load('calibration.mat');
% 
% 
% world = imread(sprintf('camera%d\\preview.png',camera));
% cameraMatrix = calibration(camera).cameraMatrix;
% distCoeffs   = calibration(camera).distCoeffs;
world = [];

if camera == 1
    imagePoints = [413.750000000000,571.250000000000;1098.87500000000,510.125000000000;1856.37500000000,660.875000000000;789.875000000000,773.375000000000];
    worldPoints = [24.955, 72.67; 24.955, 79.98; 16.295, 82.58; 16.295, 71.86];
    xInterval = 14:25;
    yInterval = 66:87;
    previewTranslation = kron([1;1;1;1],[2*100,-25*100]);
    scaling = 40;
    
elseif camera == 2
    
    imagePoints = [683.375000000000,810.875000000000;805.0625  451.4375;1257.87500000000,408.875000000000;1307.75000000000,842.750000000000];
    worldPoints = [9.125, 28.79; 23.325, 28.79; 24.955, 32.85; 7.675, 32.85];

    xInterval = 0:33;
    yInterval = 20:33;
    previewTranslation = kron([1;1;1;1],[0*100,-5*100]);
    scaling = 30;
    
%     imagePoints = [1257.87500000000,408.875000000000;1307.75000000000,842.750000000000; 1802.18750000000,931.437500000001; 1611.68750000000,496.437500000000];
%     worldPoints = [24.955, 32.85; 7.675, 32.85; 8.495 ,36.25 ; 24.955, 36.25];
%     xInterval = 0:25;
%     yInterval = 33:36;
%     previewTranslation = kron([1;1;1;1],[0*100,-5*100]);
%     scaling = 30;
    
    
elseif camera == 3
    
    imagePoints = [1514.85740514076,683.266217870257;66.0079559363524,644.288928463997;387.407516542992,866.483441772297;1516.08798863984,838.098635750109];
    ab = 11.01;
    bc = 7.27;
    cd = 6.18;
    ad = 7.47;
    ac = 12.14;
    bd = 10.02;
    
    shiftAngle = 9;
    
    wa = [24.955, 9.15];
    wb = wa + ab*[-sin(0-shiftAngle*pi/180), -cos(0-shiftAngle*pi/180)];
    wc = wa + ac*[-sin( (36.1899-shiftAngle)*pi/180), -cos((36.1899-shiftAngle)*pi/180)];
    wd = wa + ad*[-sin( (36.1899 + 24.5404-shiftAngle)*pi/180), -cos((36.1899+24.5404-shiftAngle)*pi/180)];
    
    worldPoints = [24.955, 9.15; wb; wc; wd];
    
    xInterval = 10:32;
    yInterval = -10:15;
    previewTranslation = kron([1;1;1;1],[0*100,5*100]);
    scaling = 50;
    
elseif camera == 4
    
    imagePoints = [1620.68750000000,1017.68750000000;1292.37500000000,1019.37500000000;1314.12500000000,373.625000000000;1460.18750000000,370.437500000000];
    worldPoints = [34, -21.55; 32.17, -21.55; 32.17, -35.35; 34, -35.35];
    
    xInterval = 31:35;
    yInterval = -45:-20;
    previewTranslation = kron([1;1;1;1],[0*100,12*100]);
    scaling = 20;
    
elseif camera == 5
    
    imagePoints = [436.343750000000,749.281250000000;688.343750000000,341.656250000000;1696.43750000000,229.062500000000;1873.90625000000,898.531250000000];
    worldPoints = [-5.485, 0; 5.485, 0; 8.545, 9.15; -8.545, 9.15];
    xInterval = -14:16;
    yInterval = -4:11;
    
    previewTranslation = 5 * 100;
    scaling = 30;
    
elseif camera == 6
    
    imagePoints = [1484.09375000000,878.093750000000;398.468750000000,926.843750000000;311.843750000000,646.906250000000;947.093750000000,631.156250000000];
    worldPoints = [-24.52, -35.56; -16.9, -35.56; -16.985, -13.9; -26.085, -13.9];
    xInterval = -26:-14;
    yInterval = -45:-13;
    previewTranslation = kron([1;1;1;1],[15*100,13*100]);
    scaling = 30;
    
elseif camera == 7
    
    imagePoints = [1714.62500000000,1029.12500000000;418.625000000000,906.875000000000;333.312500000000,793.812500000000;1852.43750000000,893.187500000000];
    worldPoints = [-21.625, -3.75; -19.845, 4.66; -25.085, 9.15; -25.085, -3.75];
    worldPoints = [-19.625, -3.75; -19.785, 4.61; -25.085, 9.15; -25.085, -3.75];
    yInterval = -4:10;
    xInterval = -28:-14;
    previewTranslation = kron([1;1;1;1],[20*100,3*100]);
    scaling = 40;
   
elseif camera == 8
    
    imagePoints = [1824,1071;443,775;1374,270.000000000000;1837,302.000000000000];
    worldPoints = [-49.43/2, 36.25 + 12.25;  -49.43/2, 36.25; 49.43/2, 36.25; 49.43/2, 36.25 + 12.25];
    
    yInterval = 36:50;
    xInterval = -30:25;
    previewTranslation = kron([1;1;1;1],[20*100,3*100]);
    scaling = 40;
    
    
end

% check wehther you need the actual transformation or just the planar
% correspondance for further camera calibration
if ~isempty(points)
    projectiveTransform = fitgeotrans(worldPoints,imagePoints, 'Projective');
    world = transformPointsInverse(projectiveTransform, points);
end


end

