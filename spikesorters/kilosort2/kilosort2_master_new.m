try:
    % prepare for kilosort execution
    addpath(genpath('{kilosort2_path}'));

    % set file path
    fpath = '{output_folder}';

    % create channel map file
    run(fullfile('{channel_path}'));

    % Run the configuration file, it builds the structure of options (ops)
    run(fullfile('{config_path}'))

    ops.trange = [0 Inf]; % time range to sort

    % preprocess data to create temp_wh.dat
    rez = preprocessDataSub(ops);

    % time-reordering as a function of drift
    rez = clusterSingleBatches(rez);

    % main tracking and template matching algorithm
    rez = learnAndSolve8b(rez);

    % final merges
    rez = find_merges(rez, 1);

    % final splits by SVD
    rez = splitAllClusters(rez, 1);

    % final splits by amplitudes
    rez = splitAllClusters(rez, 0);

    % decide on cutoff
    rez = set_cutoff(rez);

    fprintf('found %d good units \n', sum(rez.good>0))

    % write the output
    timestamps = rez.st3(:,1); %time
    labels = rez.st3(:,2); %cluster

    timestamps_file = fopen([fpath filesep 'timestamps.txt'],'w');
    fprintf('%g\n', timestamps)
    fclose(timestamps_file)

    labels_file = fopen([fpath filesep 'labels.txt'],'w');
    fprintf('%g\n', labels)
    fclose(labels_file)

    % we wouldn't need to do the following if we had the recording extractor in get_result_from_folder()
    samplefreq_file = fopen([fpath filesep 'samplefreq.txt'],'w');
    fprintf('%g\n', {sample_rate})
    fclose(samplefreq_file)
catch
    fprintf('----------------------------------------');
    fprintf(lasterr());
    quit(1);
end
quit(0);
