function P = RecogniseFace(I,featureType,classifierName)
% FACE RECOGNITION FUNCTION
%   Returns a matrix representing the people present in an image
%   Author: Joaquin Perez-Lapillo. City, University of London
%   Inputs required:
%       I              : an image in formats {".jpg",".jpeg"}
%       featureType    : {"SURF","HOG","NIL"}
%       classifierName : {"SVM","MLP","RF","CNN"}
%   Output: matrix "P", containing
%       P(:,1) : ID of the person
%       P(:,2) : central x position of the face
%       P(:,3) : central y position of the face

% ------------------------------ Function ------------------------------ %

% Read image
I = imread(I);

% Transform image into grayscale
Ig = rgb2gray(I);

% Face Detector initial parameters
myFaceDetector = vision.CascadeObjectDetector('ClassificationModel','FrontalFaceCART');
myFaceDetector.MergeThreshold = 8; % for supressing false detections

% Find BBOXes on the image
BBOX = myFaceDetector(Ig);

%B = insertObjectAnnotation(I,'rectangle',BBOX,'Face');
%imshow(B),title('Detected Faces');

% Iterating over faces to classify them
N = size(BBOX,1); % count of faces
P = zeros(N,3); % initialise "P" matrix
scale = [100 100]; % for resizing images (SVM, MLP, RF)
nnscale = [227 227]; % for resizing images (CNN)

% Load requested model
if strcmp(featureType,'SURF') && strcmp(classifierName,'SVM')
    load('SURF_SVM_faceClassifier.mat');
elseif strcmp(featureType,'HOG') && strcmp(classifierName,'SVM')
    load('HOG_SVM_faceClassifier.mat');
elseif strcmp(featureType,'HOG') && strcmp(classifierName,'MLP')
    load('HOG_MLP_faceClassifier.mat');
elseif strcmp(featureType,'SURF') && strcmp(classifierName,'MLP')
    load('SURF_MLP_faceClassifier.mat');
    load('SURF_MLP_bag.mat');
elseif strcmp(featureType,'SURF') && strcmp(classifierName,'RF')
    load('SURF_RF_faceClassifier.mat');
    load('SURF_MLP_bag.mat');
elseif strcmp(featureType,'NIL') && strcmp(classifierName,'CNN')
    load('CNN_faceClassifier.mat');
else
    disp('Invalid input. Use a valid format.')
end

% Load index-label table
load('labels.mat');

% Face detection process using pre-trained classifiers
for b = 1:N
    
    % using RGB images for CNN and grayscale for others
    if strcmp(classifierName,'CNN')
        F = imcrop(I,BBOX(b,:));
        Fr = imresize(F,nnscale);
    else
        F = imcrop(Ig,BBOX(b,:));
        Fr = imresize(F,scale);       
    end
    
    % extract features and predict for each classifier 
    if strcmp(featureType,'HOG') && strcmp(classifierName,'SVM')
        HoGFeatures = extractHOGFeatures(Fr);
        [labelIdx, ~] = predict(HOG_SVM_faceClassifier, HoGFeatures);
        P(b,1) = str2double(labelIdx);
    elseif strcmp(featureType,'HOG') && strcmp(classifierName,'MLP')
        HoGFeatures = extractHOGFeatures(Fr);
        xTest = HoGFeatures';
        outputTest = net(xTest);
        maxValue = max(outputTest);
        [labelIdx, ~] = find(outputTest == maxValue);
        P(b,1) = labelsNum(labelIdx);
    elseif strcmp(featureType,'SURF') && strcmp(classifierName,'MLP')
        featureVectorTest = encode(bag,Fr);
        xTest = featureVectorTest';
        outputTest = net(xTest);
        maxValue = max(outputTest);
        [labelIdx, ~] = find(outputTest == maxValue);
        P(b,1) = labelsNum(labelIdx);
    elseif strcmp(featureType,'SURF') && strcmp(classifierName,'RF')
        featureVectorTest = encode(bag,Fr);
        [labelIdx, ~] = predict(SURF_RF_faceClassifier, featureVectorTest);
        P(b,1) = str2double(labelIdx);
    elseif strcmp(featureType,'SURF') && strcmp(classifierName,'SVM')
        [labelIdx, ~] = predict(SURF_SVM_faceClassifier, Fr);
        P(b,1) = str2double(SURF_SVM_faceClassifier.Labels(labelIdx));
    else
        [labelIdx, ~] = classify(JPLnet,Fr);
        P(b,1) = str2double(cellstr(labelIdx));
    end
    
    % register the central coordinates into the "P" matrix
    P(b,2) = int64(BBOX(b,1) + (BBOX(b,3)/2)); % central x point
    P(b,3) = int64(BBOX(b,2) + (BBOX(b,4)/2)); % central x point
end
disp(P);
end

