%% ---- CV02: Get faces from images ---- %%
% finding and cropping faces to then usen them in a classifier
clear
clc

%% define directories
pathIn = 'C:\Users\Joaquin\Documents\MscDataScience\02_Spring_term\01_Computer_Vision\Coursework\dataorder\individual\';
pathOut = 'C:\Users\Joaquin\Documents\MscDataScience\02_Spring_term\01_Computer_Vision\Coursework\faces\';
folders = dir(pathIn);

%% define Cascade Object Detector for face detection
myFaceDetector = vision.CascadeObjectDetector('ClassificationModel','FrontalFaceCART');
myFaceDetector.MinSize = [100 100]; % for selecting only relevant boxes
myFaceDetector.MaxSize = [700 700]; % for selecting only relevant boxes
myFaceDetector.MergeThreshold = 10; % for supressing false detections
scale = [100 100]; % for resizing images

%% iterating over folders
disp(strcat('---- ',datestr(now,'HH:MM:SS'),' Starting face detection process ----'));
for i = 3:size(folders,1)
    % get folder name
    folderName = folders(i).name;
    filePath = strcat(pathIn,folderName);
    % create a image dataset
    imgDatabase = imageSet(filePath,'recursive');
    % loop over images inside each folder
    for j = 1:size(imgDatabase.ImageLocation,2)
        % select image
        I = read(imgDatabase(1),j);
        % transform into grayscale
        Ig = rgb2gray(I);
        % using FaceDetector to get BBOX
        BBOX = myFaceDetector(Ig);
        % conditionals on what BBOX returns:
        if isempty(BBOX)
            continue % go to next iteration if no face is found
        elseif size(BBOX,1) > 1
            [val, idx] = max(BBOX(:,3));
            BBOX = BBOX(idx,:); % keep only the biggest box assuming that it is the face
        end
        % crop the image to the BBOX
        F = imcrop(Ig,BBOX);
        % resize image to a standard size
        Fr = imresize(F,scale);
        % save it in a new directory
        fileName = strcat(pathOut,folderName,'\','IMG_',num2str(j),'.jpg');
        imwrite(Fr,fileName);
    end
    disp(strcat('Folder: ',folderName,' ,done')); 
end
disp(strcat('---- ',datestr(now,'HH:MM:SS'),' Face detection process has finished ----'));

%% Special code for individuals with few faces detected
% after running the above code, we will do additional work to increase the training set 
faceDatabase = imageSet('C:\Users\Joaquin\Documents\MscDataScience\02_Spring_term\01_Computer_Vision\Coursework\faces','recursive');
labels = {faceDatabase.Description}; % display all labels on one line
imgCount = [faceDatabase.Count]; % show the corresponding count of images
% plot an histogram
figure;
hist(imgCount)
title('Number of faces detected per individual')
% selecting the ones with less than 100 faces detected
list = find(imgCount < 100);
listLabels = str2double(labels((list)));

%% Body detector 
myBodyDetector = vision.CascadeObjectDetector('ClassificationModel','UpperBody');
myBodyDetector.MinSize = [60 60];
myBodyDetector.MergeThreshold = 10;
% Use ROI for face detector and setting parameters to default
myFaceDetector = vision.CascadeObjectDetector('ClassificationModel','FrontalFaceCART');
myFaceDetector.UseROI = true;
myFaceDetector.MinSize = [];
myFaceDetector.MaxSize = [];
myFaceDetector.MergeThreshold = 4;

scale = [100 100]; % for resizing images

%% Find body, then face
for i = 1:size(listLabels,2)
    % get folder name
    folderName = num2str(listLabels(i));
    filePath = strcat(pathIn,folderName);
    % create a image dataset
    imgDatabase = imageSet(filePath,'recursive');
    % loop over images inside each folder
    for j = 1:size(imgDatabase.ImageLocation,2)
        % select image
        I = read(imgDatabase(1),j);
        % transform into grayscale
        Ig = rgb2gray(I);
        % using upper body detector
        B_BBOX = myBodyDetector(Ig);
         % conditionals on what B_BBOX returns:
        if isempty(B_BBOX)
            continue % go to next iteration if no face is found
        elseif size(B_BBOX,1) > 1
            [val, idx] = max(B_BBOX(:,3));
            B_BBOX = B_BBOX(idx,:); % keep only the biggest box assuming that it is the body
        end       
        % using FaceDetector to get BBOX with a region of interest (ROI)
        BBOX = myFaceDetector(Ig,B_BBOX);
        % conditionals on what BBOX returns:
        if isempty(BBOX)
            continue % go to next iteration if no face is found
        elseif size(BBOX,1) > 1
            [val, idx] = max(BBOX(:,3));
            BBOX = BBOX(idx,:); % keep only the biggest box assuming that it is the face
        end
        % crop the image to the BBOX
        F = imcrop(Ig,BBOX);
        % resize image to a standard size
        Fr = imresize(F,scale);
        % save it in a new directory
        fileName = strcat(pathOut,folderName,'\','IMG_',num2str(j),'.jpg');
        imwrite(Fr,fileName);
    end
    disp(strcat('Folder: ',folderName,' ,done')); 
end



