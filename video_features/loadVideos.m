
% THIS WILL LIKELY BE REPLACED DURING FULL SYSTEM TESTING
% NEED FILE NAME FORMAT TO GET LABELS
%}

function [videos, labels] = loadVideos()
    filePath = "../videos/Yubo/*.mp4";
    files = dir(filePath);
    
    videos = zeros(30, 200,200);
    labels = strings(height(files(:)),1);


    for i = 1:height(files)
        v = VideoReader(strcat(files(i).folder, '/', files(i).name));
        label = split(files(i).name, "_");
        num = split(label(2), ".");
        labels(i) = label(1);

        vHeight = v.Height;
        vWidth = v.Width;
        s = struct("cdata", zeros(vHeight, vWidth, 3, "uint8"), 'colormap', []);
        
        k = 1;
        while hasFrame(v)
            s(k).cdata = readFrame(v);
            
            %imshow(s(k).cdata)
            [binimg, roi] = lab_10(s(k).cdata);
            
            binimg = imresize(binimg, [200,200]);

            k=k+1;

            writeTo = "../bin_Images/"+labels(1)+"/" +labels(1)+num(1);
            if ~isfolder(writeTo)
                mkdir("../bin_Images/"+ labels(1), labels(1)+num(1));
            end    
            imwrite(binimg, writeTo +"/"+labels(1)+"_"+num(1) +"_"+ k+".png")

            
        end

        % P = roipoly(mov(1).cdata);
        % ptr = find(P);
    end    

end