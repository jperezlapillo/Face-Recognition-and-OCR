function output = detectNumImg(I)
% OPTICAL CHARACTER RECOGNITION FUNCTION
%   Returns numbers detected in an individual image
%   Author: Joaquin Perez-Lapillo. City, University of London
%   Input: uint8 image
%   Output: string containing the recognised number on the image


% ------------------------------ Function ------------------------------ %

% initialise blob analyser
Hblob = vision.BlobAnalysis('MaximumCount', 500);

% set scale to 25% of the standard image size used for development
scale = [1008 756];

% check if the image is rotated. If it is, rotate back
if size(I,1) < size(I,2)
    I = imrotate(I,270);
end    

% resize to 25%
Ir = imresize(I,scale);

% transform to grayscale
Ig = rgb2gray(Ir);

% increasing brightness according to mean value (slope=-1, constant=255)
%   (frames from videos have a black frame around the actual image, so we 
%    filter them by setting a threshold)
meanBright = mean(mean(Ig));
if meanBright > 100
    Ig = Ig + (255 + -1*meanBright);
end

% transform image to the black and white inverse
Ib = imcomplement(imbinarize(Ig));

% finding regions of interest using BlobAnalysis
[~, ~, BBOX] = step(Hblob, Ib);

% keeping only relevant BBOXes
BBOX = BBOX(BBOX(:,3) > 10,:); % lenght greater than 10 pixels
BBOX = BBOX(BBOX(:,3) < 50,:); % lenght less than 50 pixels
ratio = double(BBOX(:,3)) ./ double(BBOX(:,4)); % width/hight ratio
BBOX = BBOX(ratio > 0.3 & ratio < 1 ,:); % keep only vertical rectangles

% filter boxes on the extremes of the image (keep internal 90%)
BBOX = BBOX(BBOX(:,1) > 75 & BBOX(:,1) + BBOX(:,3) < size(Ir,2) - 75 & ...
    BBOX(:,2) > 100 & BBOX(:,2) + BBOX(:,4) < size(Ir,1) - 100,:);

% increasing size of boxes for a better object detection
iBBOX = zeros(size(BBOX));
iBBOX(:,1) = BBOX(:,1) - 2; % moving starting x to the left
iBBOX(:,2) = BBOX(:,2) - 2; % moving up starting y
iBBOX(:,3) = BBOX(:,3) + 4; % increasing width
iBBOX(:,4) = BBOX(:,4) + 4; % increasing lenght

% transforming the black and white boolean image to uint8
Ib = im2uint8(Ib);

% pass image to Matlab OCR function to detect numbers on the image
objects = ocr(Ib, iBBOX,'CharacterSet', '0123456789', 'TextLayout','Word');

% filtering out objects with no number detected
for o = size(objects,1):-1:1
    if isnan(str2double(objects(o,1).Text))
        objects(o,:) = [];
    end
end

% removing low-confidence numbers detected by OCR 
% threshold: 80% of the object with the highest confidence
maxConf = max([objects.WordConfidences]);
objects = objects([objects.WordConfidences] > 0.8*maxConf); 

% merge numbers detected into one single value
if size(objects,1) > 0
    numArray = zeros(1,size(objects,1));
    for o = 1:size(objects,1)
        numArray(1,o) = str2double(objects(o,1).Text);
    end
    output = strings(1,1);
    for n = 1:size(numArray,2)
        output = strcat(output,num2str(numArray(1,n)));
    end
else
    output = [];
end
end

