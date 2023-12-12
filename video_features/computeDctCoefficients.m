function computeDctCoefficients(videoPath, outputName)
    % computeDCTCoefficients Given an input video path, calculates the DCT
    % for each frame of the video and stores these coefficients to a file

    v = VideoReader(videoPath);
    firstFrame = readFrame(v);

    if size(firstFrame, 3) == 3
        firstFrame = rgb2gray(firstFrame);
    end

    numCoeffs = numel(dct2(firstFrame));
    dctCoeffs = zeros(30, numCoeffs, 'single'); % use single precision

    v = VideoReader(videoPath);

    for frameCount = 1:min(30, v.NumFrames)
        frame = readFrame(v);

        if size(frame, 3) == 3
            frame = rgb2gray(frame);
        end

        dctFrame = dct2(frame);

        dctCoeffs(frameCount, :) = single(dctFrame(:)');
    end

    tensorInput = reshape(dctCoeffs, [30, size(firstFrame, 1), size(firstFrame, 2), 1]);

    split_string = strsplit(outputName, '_');
    outputDir = 'testing';
    if str2double(split_string{2}) <= 20
        outputDir = 'training';
    end

    outputFile = fullfile(outputDir, sprintf('%s.mat', outputName));
    save(outputFile, 'tensorInput');
end
