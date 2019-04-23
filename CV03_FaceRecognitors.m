%% ---- CV02: Face Recognitor ---- %%
% creating different face recognitors using individual images
% models trained in this script: SVM, MLP, and RF

clear
clc

%% import data, preparation and train/test split
% import database
faceDatabase = imageSet('C:\Users\Joaquin\Documents\MscDataScience\02_Spring_term\01_Computer_Vision\Coursework\faces','recursive');
labels = {faceDatabase.Description}; % display all labels on one line
imgCount = [faceDatabase.Count]; % show the corresponding count of images

% data preparation
minSetCount = min(imgCount); % minimum number of images for each label
faceDatabase = partition(faceDatabase, minSetCount, 'randomize'); % randomly reduce the sets to the minimum

% training / validation / test split
[training,test] = partition(faceDatabase,[0.7 0.3]); % 63 for training, 27 for test

%% ---- 1. SURF - SVM ----
% 1.1. extract SURF features using bag
bag = bagOfFeatures(training);

% 1.2. train SVM classifier
SURF_SVM_faceClassifier = trainImageCategoryClassifier(training, bag);

% 1.3. save classifier
save('SURF_SVM_faceClassifier.mat','SURF_SVM_faceClassifier')

% 1.3. evaluate performance
% training
SURF_SVM_confMatrixTrain = evaluate(SURF_SVM_faceClassifier, training);
SURF_SVM_accTrain = mean(diag(SURF_SVM_confMatrixTrain));

% test
SURF_SVM_confMatrixTest = evaluate(SURF_SVM_faceClassifier, test);
SURF_SVM_accTest = mean(diag(SURF_SVM_confMatrixTest ));

%% ---- 2. HOG - SVM ----
% 2.1. extract HOG features
% size of the HOG feature array with 100x100 images: 4356
% size of the HOG feature array with 50x50 images: 900

arraySize = 4356;
HoGtrainingFeatures = zeros(size(training,2)*training(1).Count,arraySize);
count = 1;
for i=1:size(training,2) % for each individual
    for j = 1:training(i).Count % for each training image of the subject
        HoGtrainingFeatures(count,:) = extractHOGFeatures(read(training(i),j)); % extract and save HOG features of the face as a row
        trainingLabel{count} = training(i).Description; % label is stored as "description"
        count = count + 1;
    end
    personIndex{i} = training(i).Description;
end

% 2.2. train an SVM classifier
HOG_SVM_faceClassifier = fitcecoc(HoGtrainingFeatures,trainingLabel);

% 2.3. save classifier
save('HOG_SVM_faceClassifier.mat','HOG_SVM_faceClassifier')

% 2.3. evaluate performance

% training
[predictionTrain, ~] = predict(HOG_SVM_faceClassifier, HoGtrainingFeatures);
predictionTrain = predictionTrain';

HOG_SVM_confMatrixTrain = confusionmat(trainingLabel, predictionTrain);
HOG_SVM_accTrain = (sum(diag(HOG_SVM_confMatrixTrain)))/(sum(sum(HOG_SVM_confMatrixTrain)));

% test
% extract features from test set
HoGtestFeatures = zeros(size(test,2)*test(1).Count,arraySize);
count = 1;
for i=1:size(test,2) % for each individual
    for j = 1:test(i).Count % for each training image of the subject
        HoGtestFeatures(count,:) = extractHOGFeatures(read(test(i),j)); % extract and save HOG features of the face as a row
        testLabel{count} = test(i).Description; % label is stored as "description"
        count = count + 1;
    end
    personIndex{i} = test(i).Description;
end

% predict
[predictionTest, ~] = predict(HOG_SVM_faceClassifier, HoGtestFeatures);
predictionTest = predictionTest';

% evaluate
HOG_SVM_confMatrixTest = confusionmat(testLabel, predictionTest);
HOG_SVM_accTest = (sum(diag(HOG_SVM_confMatrixTest)))/(sum(sum(HOG_SVM_confMatrixTest)));

%% ---- 3. SURF - MLP ----
% 3.1. extract SURF features using bag
bag = bagOfFeatures(training);
featureVector = encode(bag,training);
xTrain = featureVector'; % transpose the feature vector
trainCount = size(training,2)*training(1).Count;

% training targets
tTrainLabels = zeros(1,trainCount);
count= 1;
for i=1:size(training,2)
    for j=1:training(i).Count
        tTrainLabels(1,count) = str2double(training(i).Description);
        count = count + 1;
    end
end

% one-hot training targets vector
tTrain = zeros(size(training,2),trainCount);

% get 1-hot target vector
count= 1;
for i=1:size(training,2)
    for j=1:training(i).Count
        tTrain(i,count) = 1;
        count = count + 1;
    end
end

% 3.2. train a MLP network
net = feedforwardnet(100, 'trainscg'); % 100 neurons in the hidden layer and training function = SCG
net = configure(net,xTrain,tTrain);
net = train(net,xTrain,tTrain); % training the network

% 3.3. save classifier
save('SURF_MLP_faceClassifier.mat','net')
save('SURF_MLP_bag.mat','bag')

% 3.4. evaluate performance
% training
outputTrain = net(xTrain);
% assign an answer to the max value of the column and store it as labels
labelsNum = str2double(labels);
for i = 1:trainCount
    [value outputTrainLabels(1,i)] = max(outputTrain(:,i));
    outputTrainLabels(1,i) = labelsNum(1,outputTrainLabels(1,i));
end
% Calculate the accuracy of the neural network on training and test:
SURF_MLP_accTrain = sum(outputTrainLabels == tTrainLabels) / trainCount;

% test
% extract features
featureVectorTest = encode(bag,test);
xTest = featureVectorTest'; % transpose the feature vector
testCount = size(test,2)*test(1).Count;
% test targets
tTestLabels = zeros(1,testCount);
count= 1;
for i=1:size(test,2)
    for j=1:test(i).Count
        tTestLabels(1,count) = str2double(test(i).Description);
        count = count + 1;
    end
end
% output
outputTest = net(xTest);
% assign an answer to the max value of the column and store it as labels
for i = 1:testCount
    [value outputTestLabels(1,i)] = max(outputTest(:,i));
    outputTestLabels(1,i) = labelsNum(1,outputTestLabels(1,i));
end
% Calculate the accuracy of the neural network on training and test:
SURF_MLP_accTest = sum(outputTestLabels == tTestLabels) / testCount;

%% ---- 4. HOG - MLP ----
% 4.1. extract HOG features (using same as for SVM)
xTrain = HoGtrainingFeatures';

% 4.2. train a MLP network
net = feedforwardnet(100, 'trainscg'); % 100 neurons in the hidden layer and training function = SCG
net = configure(net,xTrain,tTrain);
net = train(net,xTrain,tTrain); % training the network

save('HOG_MLP_faceClassifier.mat','net')
% 4.3. evaluate performance
% training
outputTrain = net(xTrain);
% assign an answer to the max value of the column and store it as labels
labelsNum = str2double(labels);
for i = 1:trainCount
    [value outputTrainLabels(1,i)] = max(outputTrain(:,i));
    outputTrainLabels(1,i) = labelsNum(1,outputTrainLabels(1,i));
end
% Calculate the accuracy of the neural network on training and test:
HOG_MLP_accTrain = sum(outputTrainLabels == tTrainLabels) / trainCount;

% test
% extract features
xTest = HoGtestFeatures'; % transpose the feature vector
% output
outputTest = net(xTest);
% assign an answer to the max value of the column and store it as labels
for i = 1:testCount
    [value outputTestLabels(1,i)] = max(outputTest(:,i));
    outputTestLabels(1,i) = labelsNum(1,outputTestLabels(1,i));
end
% Calculate the accuracy of the neural network on training and test:
HOG_MLP_accTest = sum(outputTestLabels == tTestLabels) / testCount;

%% ---- 5. Random Forest - SURF ----
% 1.2. train RF classifier
SURF_RF_faceClassifier = TreeBagger(100, featureVector, tTrainLabels');

% 1.3. save classifier
save('SURF_RF_faceClassifier.mat','SURF_RF_faceClassifier')

% 1.3. evaluate performance
% training
SURF_RF_outputTrain = str2double(predict(SURF_RF_faceClassifier, featureVector));
SURF_RF_confMatrixTrain = confusionmat(tTrainLabels', SURF_RF_outputTrain);
SURF_RF_accTrain = (sum(diag(SURF_RF_confMatrixTrain)))/(sum(sum(SURF_RF_confMatrixTrain)));

% test
SURF_RF_outputTest = str2double(predict(SURF_RF_faceClassifier, featureVectorTest));
SURF_RF_confMatrixTest = confusionmat(tTestLabels', SURF_RF_outputTest);
SURF_RF_accTest = (sum(diag(SURF_RF_confMatrixTest)))/(sum(sum(SURF_RF_confMatrixTest)));





