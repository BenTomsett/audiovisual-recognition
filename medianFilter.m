% 
% function lab_10()
%     image = imread('cameraman.tif');
%     %imshow(image);
% 
%     %Easier way to create filter of identical values
%     kernel = zeros(5,5);
%     image = imnoise(image, "salt & pepper", 0.02);
%     imshow(image)
%     %fimg = medianFilter(image, kernel);
%     %imshow(fimg, []);
% 
%     medfiltImg = medfilt2(image, [5,5])
%     imshow(medfiltImg)
% end


function medimg = medianFilter(img, window)
[x, y] = size(img); 
[a, b] = size(window);
medimg = zeros(x, y);

minX = ceil(-a/2);
maxX = floor(a/2);
minY = ceil(-b/2);
maxY = floor(b/2);

for i = 1:x
    for j = 1:y

        if i + minX < 1 && j + minY <1
            window =img(1:i+maxX, 1:j+maxY);

        elseif i + minX < 1 && j + maxY > y
            window =img(1:i+maxX, j+minY:y);

        elseif i + maxX >x && j + maxY >y
            window =img(i+minX:x, j+minY:y);
        
        elseif i + maxX >x && j + minY <1
            window =img(i+minX:x, 1:j+maxY);

        elseif i + minX < 1
            window =img(1:i+maxX, j+minY: j+maxY);

        elseif j + minY < 1
            window =img(i+minX:i+maxX, 1:j+maxY);

        elseif j + maxY > y
            window =img(i+minX:i+maxX, j+minY:y);

        elseif i + maxX >x
            window =img(i+minX:x, j+minY: j+maxY);

        else
            window = img(i+minX:i+maxX, j+minY: j+maxY);
        end
        medimg(i,j) = median(window(:));
    end
end
medimg = uint8(medimg);
end
