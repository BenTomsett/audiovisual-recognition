
function Lab_9()
    image = imread('cameraman.tif');
    %imshow(image);
    %kernel = [0.1111 0.1111 0.1111; 0.1111 0.1111 0.1111; 0.1111 0.1111 0.1111];
    
    %Easier way to create filter of identical values
    kernel = ones(3,3) * 0.1111;
    fimg = linFilter(image, kernel);
    imshow(fimg, []);
end

function filteredImg = linFilter(img, kernel)
[x, y] = size(img);
[a, b] = size(kernel);
filteredImg = zeros(x, y);
for i = 1:x
    for j = 1:y
        g = zeros(a,b);
        for m = ceil(-a/2):floor(a/2)
            for n = ceil(-b/2):floor(b/2)
                if (j-n < 1 || j-n > y) && (i-m < 1 || i-m > x) 
                    g(ceil(a/2)+m,ceil(b/2)+n) = img(i, j)*kernel(ceil(a/2)+m,ceil(b/2)+n);
                elseif i-m <1 || i-m > x 
                    g(ceil(a/2)+m,ceil(b/2)+n) = img(i, j-n)*kernel(ceil(a/2)+m,ceil(b/2)+n);
                elseif j-n < 1 || j-n > y
                    g(ceil(a/2)+m,ceil(b/2)+n) = img(i-m, j)*kernel(ceil(a/2)+m,ceil(b/2)+n);
                else
                    g(ceil(a/2)+m,ceil(b/2)+n) = img(i-m, j-n)*kernel(ceil(a/2)+m,ceil(b/2)+n);
                end
            end
        end
        filteredImg(i,j) = mean(mean(g));
    end
end
%filteredImg = cat(3, filteredImg, filteredImg, filteredImg);
filteredImg = uint8(filteredImg);
end


