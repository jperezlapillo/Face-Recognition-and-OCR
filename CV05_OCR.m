%% ---- CV05: OCR ---- %%
% developing an object character recognitor to recognise numbers on
% individual images or videos

clear;clc

% development path
path = 'C:\Users\Joaquin\Documents\MscDataScience\02_Spring_term\01_Computer_Vision\Coursework\ocr\';
files = dir(path);
images = {files(~[files.isdir]).name};

% set rescale constant to 25%
scale = [1008 756];

% initialise blob analyser
Hblob = vision.BlobAnalysis('MaximumCount', 500);

for i = 1:87
    % read image
    I = imread(strcat(path,cell2mat(images(i))));
    % resize to 25%
    Ir = imresize(I,scale);
    % transform to grayscale
    Ig = rgb2gray(Ir);
    % get average brightness value of the image
    meanBright = mean(mean(Ig));
    % increase brightness according to mean brightness (slope=-1, constant=255)
    Ii = Ig + (255 + -1*meanBright);
    % black and white
    Ib = imcomplement(imbinarize(Ii));
    % find area using blob analyser
    [areas, centroids, BBOX] = step(Hblob, Ib);
    % keep only relevant BBOXes
    BBOX = BBOX(BBOX(:,3) > 10,:); % lenght greater than 10 pixels
    BBOX = BBOX(BBOX(:,3) < 50,:); % lenght less than 50 pixels
    ratio = double(BBOX(:,3)) ./ double(BBOX(:,4)); % ratio between width and hight
    BBOX = BBOX(ratio > 0.3 & ratio < 1 ,:); % keep vertical rectangles
    % filter boxes on the extremes of the image (keep internal 90%)
    BBOX = BBOX(BBOX(:,1) > 75 & BBOX(:,1) + BBOX(:,3) < size(Ir,2) - 75 & ...
        BBOX(:,2) > 100 & BBOX(:,2) + BBOX(:,4) < size(Ir,1) - 100,:);
    % increase size of boxes
    iBBOX = zeros(size(BBOX));
    iBBOX(:,1) = BBOX(:,1) - 2; % moving starting x to the left
    iBBOX(:,2) = BBOX(:,2) - 2; % moving up starting y
    iBBOX(:,3) = BBOX(:,3) + 4; % increasing width
    iBBOX(:,4) = BBOX(:,4) + 4; % increasing lenght
    % display BBOX on image (COMMENT OUT THIS SECTION WHEN IN PRODUCTION)
    Ib = im2uint8(Ib);
    B = insertObjectAnnotation(Ir,'rectangle',iBBOX,'BBOX');
    figure; imshow(B),title('Detected objects');
    % pass OCR to detect numbers
    objects = ocr(Ib, iBBOX,'CharacterSet', '0123456789', 'TextLayout','Word');
    % filtering the objects with no number detected
    for o = size(objects,1):-1:1
        if isnan(str2double(objects(o,1).Text))
            objects(o,:) = [];
        end
    end
    % removing low-confidence numbers detected by OCR
    % threshold: 80% of the object with the highest confidence
    maxConf = max([objects.WordConfidences]);
    objects = objects([objects.WordConfidences] > 0.8*maxConf);
    % merge numbers detected into one single integer
    if size(objects,1) > 0
        numArray = zeros(1,size(objects,1));
        for o = 1:size(objects,1)
            numArray(1,o) = str2double(objects(o,1).Text);
        end
        output = strings(1,1);
        for n = 1:size(numArray,2)
            output = strcat(output,num2str(numArray(1,n)));
        end
        disp(strcat('Image ',num2str(i),'. Number detected: ',output));
    else
        % in case of no numbers detected, return a message
        disp(strcat('Image ',num2str(i),'. No numbers detected'));
    end
    %pause;
end








