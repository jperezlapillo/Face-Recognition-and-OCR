%% ---- CV01: Processing videos ---- %%
% iterating over individual folders to get images from videos

clear
clc

%% define directory
path = 'C:\Users\Joaquin\Documents\MscDataScience\02_Spring_term\01_Computer_Vision\Coursework\dataorder\individual\';
folders = dir(path);

%% iterating over folders
n = 30; % number of frames to save
disp(strcat('---- ',datestr(now,'HH:MM:SS'),' Starting extraction of frames from videos ----'));
for i = 3:size(folders,1) % first two elements seem to be temp files
    % get folder name
    folderName = folders(i).name;
    filePath = strcat(path,folderName);
    % get list of videos on folder
    videos = dir(fullfile(filePath, '*.mp4'));
    % iterate over videos to save frames as images
    for j = 1:size(videos,1)
        videoReader = VideoReader(strcat(path,folderName,'\',videos(j).name));
        images = read(videoReader);
        for k = 1:n
            I = images(:,:,:,k);
            fileName = strcat(path,folderName,'\','IMG_FrameFromVideo_',num2str(j),'_',num2str(k),'.jpg');
            imwrite(I,fileName);
        end
    end
    disp(strcat('Folder: ',folderName,' ,done'));    
end
disp(strcat('---- ',datestr(now,'HH:MM:SS'),' Extraction of frames from videos has ended ----'));






