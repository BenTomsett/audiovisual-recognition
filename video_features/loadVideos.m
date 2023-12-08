%{
% Read in all data files
% THIS WILL LIKELY BE REPLACED DURING FULL SYSTEM TESTING
% NEED FILE NAME FORMAT TO GET LABELS
%}
function [videos, labels] = loadVideos(filePath)
    files = dir(filePath);
    videos = size(length(files));
    labels = size(length(files));
    for i = 1:length(files)
        v = videoReader(files(i));
        
        s = struct('cdata', zeros(v.Height, v.Width, 3, 'uint8'), 'colormap', []);
        videos(i) = s;
    end    

end