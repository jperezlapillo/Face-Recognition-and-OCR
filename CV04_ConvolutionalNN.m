%% ---- CV04: ConvolutionalNN ---- %%
% creating a CNN implementation using Alexnet

clear
clc

%% Image preprocessing before using alexnet
% We first need to use an imageDatastore structure
% Alexnet requires a certain size of the images, and the use of RGB
imageDatastore = imageDatastore('C:\Users\Joaquin\Documents\MscDataScience\02_Spring_term\01_Computer_Vision\Coursework\faces',...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% train/test split using the same number of images for training and test
[trainingImages,validationImages,testImages] = splitEachLabel(imageDatastore,0.7,0.15,'randomized');

% standard input size
netInputSize = [227 227];

% creating an augmenter object to apply to images (using Matlab example
% values)
augmenter = imageDataAugmenter( ...
    'RandRotation',[-10,10], ...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3],...
    'RandYReflection', true);

% applying augmentation to the dataset
Training = augmentedImageDatastore(netInputSize,trainingImages,...
    'ColorPreprocessing','gray2rgb','DataAugmentation',augmenter);
Validation = augmentedImageDatastore(netInputSize,validationImages,...
    'ColorPreprocessing','gray2rgb','DataAugmentation',augmenter);
Test = augmentedImageDatastore(netInputSize,testImages,...
    'ColorPreprocessing','gray2rgb','DataAugmentation',augmenter);

%% Network initialisation
% Load pre-trained net (Alexnet)
net = alexnet;
% Remove fully connected and classification layers and replace
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(trainingImages.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];
% Set network options
miniBatchSize = 10;
numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',6,...
    'InitialLearnRate',1e-4,...
    'Verbose',false,...
    'Plots','training-progress',...
    'ValidationData',Validation,...
    'ValidationPatience',2,...
    'ValidationFrequency',numIterationsPerEpoch);

%% Training network
[JPLnet, netInfo] = trainNetwork(Training,layers,options);

% save net for face recognition
save('CNN_faceClassifier_Aug.mat','JPLnet')

%% Performance
% preformance on training and validation sets
CNNtrainAcc = netInfo.TrainingAccuracy(end)/100;
CNNvalAcc = netInfo.ValidationAccuracy(netInfo.ValidationAccuracy > 0);
CNNvalAcc = CNNvalAcc(end)/100;

% performance on test set
testLabels = testImages.Labels;
prediction = classify(JPLnet,Test);
CNNtestAcc = sum(prediction == testLabels)/numel(testLabels);













