function output = detectNum(filename)
% OBJECT CHARACTER RECOGNITION FUNCTION
%   Returns the number that a person is holding in an image or a video
%   Author: Joaquin Perez-Lapillo. City, University of London
%   Input: a filename with the following extensions
%       Image: {".jpg",".jpeg"}
%       Video: {".mp4",".mov"}
%   Output: double, representing the number that the person is holding

% ------------------------------ Function ------------------------------ %

% check if the input file is a video or an image
if endsWith(filename, '.mp4') || endsWith(filename, '.mov')
    % read video
    videoReader = VideoReader(filename);
    % save images as an object
    images = read(videoReader);
    % define frames to be extracted
    frames = 10:40;
    % initialise output array
    outputArr = strings(1,size(frames,2));
    % iterate over selected frames
    for k = frames
        % get the frame
        I = images(:,:,:,k);
        % call detectNumImg function on the image to get the number
        num = detectNumImg(I);
        % save number into output array
        if size(num,1) > 0
            outputArr(1,k) = num;
        end
    end
    % transform output array into numeric
    outputArr = str2double(outputArr);
    % final output: take majority vote
    output = mode(outputArr);
    
elseif endsWith(filename, '.jpg') || endsWith(filename, '.jpeg')
    % read image
    I = imread(filename);
    % call detectNumImg function on the image to get the number
    output = str2double(detectNumImg(I));
    
else
    disp('Invalid input. Use a valid format.');
    
end
end

