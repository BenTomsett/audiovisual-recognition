function computeDctCoefficients(videoPath, outputName)
    % computeDCTCoefficients Given an input video path, calculates the DCT
    % in 8x8 blocks for each frame of the video and stores these coefficients to a file

    v = VideoReader(videoPath);
    firstFrame = readFrame(v);

    if size(firstFrame, 3) == 3
        firstFrame = rgb2gray(firstFrame);
    end

    [height, width] = size(firstFrame);
    numBlocksH = ceil(height / 8);
    numBlocksW = ceil(width / 8);

    dctCoeffs = zeros(30, numBlocksH * 8, numBlocksW * 8, 'single'); % Adjusted size for 8x8 blocks

    v = VideoReader(videoPath);

    for frameCount = 1:min(30, v.NumFrames)
        frame = readFrame(v);

        if size(frame, 3) == 3
            frame = rgb2gray(frame);
        end

        dctFrame = zeros(size(frame), 'single');

        % Process each 8x8 block
        for i = 1:numBlocksH
            for j = 1:numBlocksW
                rowStart = (i-1)*8 + 1;
                rowEnd = min(i*8, height);
                colStart = (j-1)*8 + 1;
                colEnd = min(j*8, width);

                block = frame(rowStart:rowEnd, colStart:colEnd);
                dctBlock = dct2(block);

                % Zero out elements not in the upper left triangle
                dctBlock = fliplr(dctBlock);
                dctBlock = triu(dctBlock);
                dctBlock = fliplr(dctBlock);

                dctFrame(rowStart:rowEnd, colStart:colEnd) = dctBlock;
            end
        end

        dctCoeffs(frameCount, :, :) = single(dctFrame);
    end

    tensorInput = reshape(dctCoeffs, [30, numBlocksH * 8, numBlocksW * 8, 1]);

    split_string = strsplit(outputName, '_');
    outputDir = 'testing';
    if str2double(split_string{2}) <= 20
        outputDir = 'training';
    end

    outputFile = fullfile(outputDir, sprintf('%s.mat', outputName));
    save(outputFile, 'tensorInput');
end
