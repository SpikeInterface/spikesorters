try
    % prepare for kilosort execution
    addpath(genpath('{hdsort_path}'), '{utils_path}/mdaio');

    %% Add 'External' functions to path:
    setup()

    outputFolder = '{output_folder}';
    mainFolder = '.';
    run(fullfile('{config_path}'));
    rawFile = '{file_name}';

    %% Get a handle onto the raw file, which will then be used as input to the sorter:
    sortingName = '{sort_name}';
    fileFormat = '{file_format}';

    %% Load params
    run(fullfile('{config_path}'))

    if (strcmp(fileFormat, 'maxwell'))
        RAW = hdsort.file.MaxWellFile(rawFile);
    elseif (strcmp(fileFormat, 'mea1k'))
        RAW = hdsort.file.BELMEAFile(rawFile);
    end

    RAW.restrictToConnectedChannels(); % This line is very important to not sort empty electrodes!

    %% Create the object that performs the sorting:
    HDSorting = hdsort.Sorting(RAW, mainFolder, sortingName)

    %% Preprocess:
    % The preprocessor loads data from the file in chunks, filters it, and saves
    % the filtered data into a new hdf5 file that is standardized for all types
    % of input data.
    % If further performs a couple of operation for each local electrode group
    % (LEG) such as spike detection, spike waveform cutting and noise
    % estimation. The result is a folder named group000x for each LEG that
    % contains the data necessary to perform the parallel parts of the sorting.
    % This implementation minimizes the number of time-consuming file access
    % operations and thus speeds up the parallel processes significantly, even
    % allowing the parallel processes to be run on a small desktop computer or
    % laptop.

    chunkSize = {chunk_size}; % This number depends a lot on the available RAM
    HDSorting.preprocess('chunkSize', chunkSize, 'forceFileDeletionIfExists', true);

    %% Sort each LEG independently:
    HDSorting.sort('sortingMode', '{loop_mode}'); % (default)
    % Alternative sorting modes are:
    % HDSorting.sort('sortingMode', 'local'); % for loop over each LEG
    % HDSorting.sort('sortingMode', 'grid'); % requires a computer grid architecture

    %% Combine the resutls of each LEG in the postprocessing step:
    HDSorting.postprocess()

    %% Export the results in an easy to read format:
    [sortedPopulation, sortedPopulation_discarded] = HDSorting.createSortedPopulation(mainFolder);

    %% When the sorting has already been run before, open the results from a file with:
    sortedPopulation = hdsort.results.Population(HDSorting.files.results);

    % save to mda
    nSpikes = 0;
    for i = 1:length(sortedPopulation.unitIDs)
        nSpikes = nSpikes + length(sortedPopulation.Units(i).spikeTrain);
    end
    spikeTimes = zeros(nSpikes, 1);
    spikeClusters = zeros(nSpikes, 1);
    firstColumn = zeros(nSpikes, 1);
    iSpike = 1;
    for i = 1:length(sortedPopulation.unitIDs)
        spikeTrain = sortedPopulation.Units(i).spikeTrain;
        nSpikesUnit = length(spikeTrain);
        spikeTimes(iSpike:iSpike+nSpikesUnit-1) = spikeTrain;
        spikeClusters(iSpike:iSpike+nSpikesUnit-1) = ...
            repelem(sortedPopulation.Units(i).ID, nSpikesUnit);
        iSpike = iSpike + nSpikesUnit;
    end
    mr_mda = [firstColumn, spikeTimes, spikeClusters];
    output_mda = fullfile(outputFolder, 'firings.mda');
    writemda(mr_mda', 'firings.mda', 'float64');
catch
    fprintf('----------------------------------------');
    fprintf(lasterr());
    quit(1);
end
quit(0);
