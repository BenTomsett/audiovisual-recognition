function [binImg, roi] = lab_10(image)
    %image = imread('coins.png');
    %figure
    %imshow(image);
    %figure
    %imhist(image)

    % Red = image(:,:,1);
    % Green = image(:,:,2);
    % Blue = image(:,:,3);
    % [yRed, x] = imhist(Red);
    % [yGreen, x] = imhist(Green);
    % [yBlue, x] = imhist(Blue);
    % subplot(1, 2, 1); 
    % 
    % plot(x, yRed, 'Red', x, yGreen, 'Green', x, yBlue, 'Blue');
    % title('Histogram of  image ', 'FontSize', 1);             
    % subplot(1, 2, 2); 
    % bar(yRed, 'Red') ,hold on , bar(yGreen, 'Green'), hold on ,bar( yBlue, 'Blue');
    % title('Histogram of Stego image ', 'FontSize', 1);

% %green = 53
% %Red upper = 75
%     Lr = (Red < 75);
%     figure
%     imshow(Lr)
%     Lb = (Blue < 50);
%     Lg = (Green < 53);
%     figure
%     imshow(Lg)
% 
%     imshow(Lr)
%     imshow(Lb)
%     imshow(Lg)
 


    %Rimage = medianFilter(image(:,:,1), zeros(5,5));
    %Gimage = medianFilter(image(:,:,2), zeros(5,5));
    %Bimage = medianFilter(image(:,:,3), zeros(5,5));
    %image  =cat(3, Rimage, Gimage, Bimage)
    
    %gimage = rgb2gray(image);

    binImg = createMaskLAB(image);
    
    structureElement = strel("line", 5, 30);
    binImg = nonLinFilter(binImg, structureElement);
    
    minx = -1;
    miny = -1;
    maxx = -1;
    maxy = -1;
    [x, y] = size(image(:,:,1));
    for i = 1:x
        for j = 1:y
            if binImg(i,j) ~= 0.0
                if (minx < 0) || (i <minx)
                    minx = i;
                    maxx = i;
                end
                if (miny < 0) || (j < miny) 
                    miny = j;
                    maxy = j;
                end
                if i > maxx
                    maxx = i;
                end
                if j > maxy
                    maxy = j;
                end
            end
        end
    end

    %roi = zeros(maxx- minx+20, maxy - miny +20);
    binImg = binImg((minx-20:maxx+20), (miny-20:maxy + 20));
    %imshow(binImg);
    %circularity(binImg, ((maxx+minx)/2-minx +20))

    roi = [minx-20, miny-20; maxx+20, maxy+20];
    %shapeFeatures = [maxx-minx, maxy-miny];
end

function edgeImg = nonLinFilter(img, strele)


erimg = imerode(img, strele);
dilimg = imdilate(img, strele);
edgeImg = dilimg - erimg;
opimg = imopen(img, strele);
climg = imclose(img, strele);

% figure
% imshow(erimg)
% figure
% imshow(dilimg)
%figure
%imshow(edgeImg);
%imshow(climg);
end

function circRatio = circularity(binImg, centrex, centrey)
    
end