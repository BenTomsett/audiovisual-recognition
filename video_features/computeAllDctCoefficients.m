function computeAllDctCoefficients(folderPath)
    if ~isfolder(folderPath)
        error('Folder not found');
    end

    files = dir(fullfile(folderPath, '**', '*.mp4'));

    for k = 1:length(files)
        stem = files(k).name(1:end-4);
        filePath = fullfile(files(k).folder, files(k).name);
        disp(['Processing ', stem]);

        computeDctCoefficients(filePath, stem);
    end
end
