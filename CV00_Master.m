%% ---- CV00: Master ----
% master file for Computer Vision coursework

clear
clc

% run matlab scripts

%run CV01_ProcessVideos.m
%run CV02_GetFaces.m
run CV03_FaceRecognitors.m
run CV04_ConvolutionalNN.m

%% Call function on a set of test images
clear;clc
G1 = RecogniseFace('IMG_8224.jpg','HOG','SVM');
G2 = RecogniseFace('IMG_8233.jpg','NIL','CNN');
G3 = RecogniseFace('IMG_8239.jpg','SURF','MLP');
G4 = RecogniseFace('IMG_8241.jpg','SURF','RF');

f1 = RecogniseFace('IMG_1.jpg','HOG','SVM');
f2 = RecogniseFace('IMG_38.jpg','NIL','CNN');
f3 = RecogniseFace('IMG_39.jpg','SURF','MLP');
f4 = RecogniseFace('IMG_40.jpg','SURF','RF');
f5 = RecogniseFace('IMG_54.jpg','SURF','SVM');
f6 = RecogniseFace('IMG_FrameFromVideo_1_17.jpg','HOG','MLP');
f7 = RecogniseFace('IMG_FrameFromVideo_4_5.jpg','HOG','SVM');


%% Call number detector
clear;clc
I1 = detectNumImg('IMG_20190128_201558.jpg',.67);
I2 = detectNumImg('IMG_3302.jpg',.67);
I3 = detectNumImg('IMG_FrameFromVideo_4_5.jpg',.50);

%% Call final OCR function
clear;clc
I1 = detectNum('IMG_1.jpg');
I38 = detectNum('IMG_38.jpg');
I39 = detectNum('IMG_39.jpg');
I40 = detectNum('IMG_40.jpg');
I54 = detectNum('IMG_54.jpg');
V1 = detectNum('IMG_1.mp4');
V41 = detectNum('IMG_41.mp4');
V69 = detectNum('IMG_69.mp4');
V71 = detectNum('IMG_71.mp4');
V77 = detectNum('IMG_77.mp4');




