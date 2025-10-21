function main_5g_positioning()
%MAIN_5G_POSITIONING Entry point for 5G NR positioning simulation and sweeps.
%
%   This script orchestrates the entire simulation, including:
%   1.  Loading configuration (PositioningConfig.m)
%   2.  Handling environment variable overrides for batch processing.
%   3.  Building a "scenario plan" to sweep across different SNR values or
%       UE positions.
%   4.  Calling 'runPositioningScenario' for each scenario in the plan.
%
%   The main simulation logic is contained within 'runPositioningScenario'.

    % --- Pre-Run MATLAB Environment Cleanup ---
    try
        % Clear all variables from the base workspace
        evalin('base','clearvars;');
    catch ME_clear
        % Warn if cleanup fails (e.g., due to permissions)
        fprintf('Warning: Failed to clear base workspace: %s\n', ME_clear.message);
    end
    close all; % Close all open figure windows
    clc;       % Clear the command window

    fprintf('Performing pre-run environment cleanup...\n');
    % Call a helper function for a more aggressive cleanup (clearing functions, cache)
    performMatlabSessionCleanup();

    % Add the current directory to the MATLAB path to ensure all helper
    % functions and classes are found.
    addpath(pwd);

    fprintf('=== 5G NR Positioning System ===\n');

    %% Configuration
    % This section loads the master configuration object and then checks
    % for any environment variables that can override the defaults. This
    % is a powerful pattern for running simulations in automated scripts
    % or on CI/CD platforms.
    try
        % Load all default simulation parameters from the PositioningConfig class
        config = PositioningConfig();
        
        % --- Optional: Channel Type Override ---
        try
            % Check for an environment variable named 'CHANNEL_TYPE'
            chOverride = string(getenv('CHANNEL_TYPE'));
            if ~isempty(chOverride) && chOverride ~= ""
                % If found, override the default channel model type
                config.channelModel.Type = lower(chOverride);
                fprintf('Channel override from env: %s\n', chOverride);
            end
        catch ME_env
            fprintf('Warning: Could not check for environment variable override: %s\n', ME_env.message);
        end
        
        % Display the final configuration (including any overrides)
        config.displayConfiguration();
    catch ME
        fprintf('Error creating configuration: %s\n', ME.message);
        fprintf('Ensure PositioningConfig.m is in the current directory.\n');
        return;
    end

    % --- Result Logging Configuration (with overrides) ---
    % Get the default logging settings from the config object
    resultLoggingCfg = cfgGet(config, 'resultLogging', struct('enable', false));
    
    % Allow environment variable to set the *output file path*
    resultLogFileEnv = getenv('POSITIONING_RESULTS_FILE');
    if ~isempty(resultLogFileEnv)
        resultLoggingCfg.enable = true;
        resultLoggingCfg.outputFile = resultLogFileEnv;
    end
    % Set default filename if not specified
    if ~isfield(resultLoggingCfg, 'outputFile') || isempty(resultLoggingCfg.outputFile)
        resultLoggingCfg.outputFile = 'positioning_sweep_results.csv';
    end
    % Set default overwrite behavior
    if ~isfield(resultLoggingCfg, 'overwrite')
        resultLoggingCfg.overwrite = true;
    end
    % Allow environment variable to control *overwriting* vs. *appending*
    overwriteEnv = getenv('POSITIONING_RESULTS_OVERWRITE');
    if ~isempty(overwriteEnv)
        overwriteVal = str2double(overwriteEnv);
        if isfinite(overwriteVal)
            resultLoggingCfg.overwrite = overwriteVal ~= 0;
        end
    end
    % Allow environment variable to force *appending* (useful for batch jobs)
    appendModeEnv = getenv('POSITIONING_RESULTS_APPEND');
    forceAppendFirst = (~isempty(appendModeEnv) && str2double(appendModeEnv) ~= 0);

    try
        % Write the final logging configuration back to the config object
        config.resultLogging = resultLoggingCfg;
    catch
        % Ignore if resultLogging is not settable (e.g., if config is not a handle class)
    end

    % --- Environment-based SNR and UKF Overrides ---
    % Allow environment variable to override *SNR*
    snrOverrideStr = getenv('POSITIONING_SNR_DB');
    snrOverrideRequested = false;
    snrOverrideVal = NaN;
    if ~isempty(snrOverrideStr)
        snrOverrideVal = str2double(snrOverrideStr);
        if isfinite(snrOverrideVal)
            snrOverrideRequested = true;
        else
            fprintf('Warning: Invalid POSITIONING_SNR_DB value: %s\n', snrOverrideStr);
        end
    end

    % Allow environment variable to override *UKF enabled/disabled*
    ukfOverrideStr = getenv('POSITIONING_UKF_ENABLE');
    ukfOverrideRequested = false;
    ukfOverrideVal = NaN;
    if ~isempty(ukfOverrideStr)
        ukfOverrideVal = str2double(ukfOverrideStr);
        if isfinite(ukfOverrideVal)
            ukfOverrideRequested = true;
        else
            fprintf('Warning: Invalid POSITIONING_UKF_ENABLE value: %s\n', ukfOverrideStr);
        end
    end

    % Get the SNR sweep configuration
    snrSweepCfg = cfgGet(config, 'snrSweep', struct('enable', false));

    % Logic to apply overrides:
    if snrSweepCfg.enable
        % If a sweep is active, ignore single-run overrides
        if snrOverrideRequested
            fprintf('Ignoring external SNR override during sweep orchestration.\n');
        end
        if ukfOverrideRequested
            fprintf('Ignoring external UKF override during sweep orchestration.\n');
        end
    else
        % If it's a single run, apply any overrides found
        if snrOverrideRequested
            config.channelConfig.SNR_dB = snrOverrideVal;
            fprintf('SNR override applied: %.1f dB\n', snrOverrideVal);
        end
        if ukfOverrideRequested
            config.enableUKFTracking = ukfOverrideVal ~= 0;
            fprintf('UKF override applied: %s\n', config.boolToString(config.enableUKFTracking));
        end
    end

    % Get a scenario tag, e.g., "Run_Set_A"
    scenarioTagEnv = strtrim(getenv('POSITIONING_SCENARIO_TAG'));

    % --- Build the Scenario Plan ---
    % This helper function creates the test matrix.
    % If snrSweep is enabled, 'scenarios' will be a struct array (e.g.,
    % {SNR=5, UKF=0}, {SNR=5, UKF=1}, {SNR=15, UKF=0}, {SNR=15, UKF=1}, ...)
    % If sweep is disabled, 'scenarios' will be a single-element struct.
    scenarios = buildScenarioPlan(config, snrSweepCfg, scenarioTagEnv, snrOverrideRequested, snrOverrideVal, ukfOverrideRequested, ukfOverrideVal);
    if isempty(scenarios)
        fprintf('No scenarios configured. Exiting.\n');
        return;
    end

    % If overwriting, delete any existing results file
    if resultLoggingCfg.enable && resultLoggingCfg.overwrite && isfile(resultLoggingCfg.outputFile)
        delete(resultLoggingCfg.outputFile);
    end

    % Take a snapshot of the base configuration. This is crucial for sweeps
    % to ensure each scenario run starts from the same clean slate.
    baseConfigSnapshot = captureConfigSnapshot(config);

    % --- Main Scenario Loop ---
    % This is the master loop. It will run once for a single simulation,
    % or N times if a sweep is configured.
    numScenarios = numel(scenarios);
    for idx = 1:numScenarios
        % Get the settings for this specific scenario
        scenario = scenarios(idx);
        
        % IMPORTANT: Reset the config to the clean snapshot
        restoreConfigFromSnapshot(config, baseConfigSnapshot);
        
        % Apply the *specific* settings for this scenario (e.g., set SNR)
        try
            config.channelConfig.SNR_dB = scenario.snr;
        catch
        end
        try
            config.enableUKFTracking = scenario.ukf;
        catch
        end

        fprintf('\n=== Running Scenario %d/%d: %s ===\n', idx, numScenarios, scenario.tag);
        fprintf('Effective SNR setting: %.1f dB; UKF tracking: %s\n', scenario.snr, config.boolToString(scenario.ukf));

        % Determine if we should append to the CSV file
        % (Always append if it's not the first run, or if forced)
        appendResults = forceAppendFirst || (idx > 1);
        
        % --- Execute the Simulation ---
        % Call the main worker function that performs one full simulation
        % run with the current configuration.
        runPositioningScenario(config, scenario.tag, resultLoggingCfg, appendResults);
    end
end

% ------------------------------------------------------------------------
% ------------------------------------------------------------------------

function runPositioningScenario(config, scenarioTag, resultLoggingCfg, appendResults)
%RUNPOSITIONINGSCENARIO Execute one full positioning simulation and log results.
%
%   This function contains the entire simulation pipeline for a *single*
%   scenario run.
%
%   Inputs:
%       config: The fully configured PositioningConfig object for this run.
%       scenarioTag: A string tag (e.g., "SNR15_UKF1") for logging.
%       resultLoggingCfg: The struct controlling CSV logging.
%       appendResults: Boolean flag (true = append to CSV, false = overwrite).

% --- Parallel Processing Setup ---
parallelPoolObj = [];
parallelFlags = struct('waveform', false, 'channel', false, 'autoShutdown', false, ...
    'maxChannelWorkers', 1, 'poolSize', 0, 'channelMemoryGuardGB', inf);
try
    % Call helper to start/get the parallel pool (parpool)
    [parallelPoolObj, parallelFlags] = setupParallelProcessing(config);
catch ME_parallel
    fprintf('Parallel initialization warning: %s\n', ME_parallel.message);
end

%% Step 1: Environment Setup
fprintf('\n=== Step 1: Environment Setup ===\n');
try
    % Call helper to load the 3D model (e.g., 'train_station.stl')
    % 'viewer' is the handle to the siteviewer window
    % 'modelBounds' and 'roomDims' are structs/vectors with the physical
    % dimensions (xmin, xmax, etc.), which are critical for placement.
    [viewer, modelBounds, roomDims] = setupEnvironment(config);
catch ME
    fprintf('Error in environment setup: %s\n', ME.message);
    return;
end

% Check if UKF (Unscented Kalman Filter) tracking is enabled
useUKF = false;
if cfgHasField(config, 'enableUKFTracking')
    val = config.enableUKFTracking;
    if (islogical(val) && val) || (ischar(val) && any(strcmpi(val, {'true', 'on', 'yes', '1'})))
        useUKF = true;
    end
end
ukf = [];
if useUKF
    %% Step 1.5: Kalman Filter Initialization
    fprintf('\n=== Step 1.5: Kalman Filter Initialization ===\n');
    % The UKF will track a 4D state: [x; y; vx; vy]
    % We use the staticRxPosition as the initial position, assuming initial velocity is zero.
    
    % Use the center of the room as a neutral initial guess
    initialState = [(modelBounds.xmin + modelBounds.xmax)/2; (modelBounds.ymin + modelBounds.ymax)/2; 0; 0];

    % Define the initial state covariance (our uncertainty)
    % We are very uncertain about position (100) and velocity (10).
    initialCovariance = diag([100, 100, 10, 10]);

    % Define the process noise (how much the velocity can change between steps)
    % This is a key tuning parameter.
    processNoise = diag([0.1, 0.1, 0.5, 0.5]);

    % Define the measurement noise (how much we trust our TDOA position measurement)
    % This is another key tuning parameter. A larger value means we trust the measurement less.
    measurementNoise = diag([10, 10]);

    % State transition function (constant velocity model)
    dt = 1.0; % Assume 1.0 second between each sweep point for the motion model
    stateTransitionFcn = @ukfStateFcn; % Points to helper function at bottom

    % Measurement function (we only *measure* position, not velocity)
    measurementFcn = @(state) ukfMeasFcn(state); % Points to helper function at bottom

    % Create the Unscented Kalman Filter object from the Sensor Fusion & Tracking Toolbox
    ukf = unscentedKalmanFilter(...
        'State', initialState, ...
        'StateCovariance', initialCovariance, ...
        'StateTransitionFcn', stateTransitionFcn, ...
        'MeasurementFcn', measurementFcn, ...
        'ProcessNoise', processNoise, ...
        'MeasurementNoise', measurementNoise);
    fprintf('Unscented Kalman Filter initialized.\n');
else
    fprintf('\n=== Step 1.5: Kalman Filter Initialization ===\n');
    fprintf('UKF tracking disabled via configuration.\n');
end

% --- Position Sweep Configuration ---
% Check if the 'positionSweep' feature is enabled in the config.
positionSweepEnabled = false;
sweepPositions = [];
samplesPerPoint = 1;
if cfgHasField(config,'positionSweep')
    sweepCfg = cfgGet(config,'positionSweep',struct());
else
    sweepCfg = struct();
end
if isfield(sweepCfg,'enable') && sweepCfg.enable
    % If enabled, generate the list of UE positions to test.
    positionSweepEnabled = true;
    numPts = max(2, sweepCfg.numPoints);
    samplesPerPoint = max(1, round(sweepCfg.samplesPerPoint));
    startPt = sweepCfg.startPoint(:).';
    endPt = sweepCfg.endPoint(:).';
    
    % Create an array of [N x 3] coordinates by linearly interpolating
    % between the start and end points.
    sweepPositions = [ ...
        linspace(startPt(1), endPt(1), numPts).', ...
        linspace(startPt(2), endPt(2), numPts).', ...
        linspace(startPt(3), endPt(3), numPts).' ];
    fprintf('\n=== Position Sweep Enabled ===\n');
    fprintf('Evaluating %d UE positions, %d trial(s) per position.\n', numPts, samplesPerPoint);
end

% If sweeping is *not* enabled, this loop will just run once.
if isempty(sweepPositions)
    sweepPositions = zeros(0,3);
    numSweepPositions = 1; % Run at least once
else
    numSweepPositions = size(sweepPositions,1);
end

% Pre-allocate a cell array to store the detailed 'struct' of results
% from every single trial at every position.
sweepResults = cell(numSweepPositions, samplesPerPoint);

% --- Main Position Sweep Loop ---
% This loop iterates over each point in the UE's path.
for sweepIdx = 1:numSweepPositions
    
    % --- UKF Step 1: PREDICT ---
    % If we are using the UKF, call the 'predict' method.
    % This moves the filter's state forward in time based on its
    % internal motion model (e.g., "if I was at X and moving at V,
    % I *predict* I will be at Y now").
    if useUKF
        predict(ukf, dt);
    end
    
    if positionSweepEnabled
        % Get the [x y z] for the current sweep point
        rxPositionOverride = sweepPositions(sweepIdx,:);
        fprintf('\n=== UE Position %d/%d: [%.2f, %.2f, %.2f] m ===\n', sweepIdx, numSweepPositions, rxPositionOverride);
    else
        % Not sweeping, so no override
        rxPositionOverride = [];
    end
    
    % --- Main Trial Loop (Monte Carlo) ---
    % This loop runs multiple times *at the same position* to average
    % out random effects (like noise).
    for trialIdx = 1:samplesPerPoint
        fprintf('\n--- Trial %d/%d for UE position %d/%d ---\n', trialIdx, samplesPerPoint, sweepIdx, numSweepPositions);

%% Step 2: Receiver Placement
fprintf('\n=== Step 2: Receiver Placement ===\n');
try
    if ~isempty(rxPositionOverride)
        % Use the position from the sweep
        rxPosition = rxPositionOverride;
        fprintf('Using sweep receiver position: [%.2f, %.2f, %.2f] m\n', rxPosition);
    elseif config.enableStaticReceiverPosition
        % Use the single hardcoded position from the config file
        rxPosition = config.staticRxPosition;
        fprintf('Using hardcoded receiver position: [%.2f, %.2f, %.2f] m\n', rxPosition);
    else
        % Generate a random position (within the room bounds)
        fprintf('Generating random receiver position...\n');
        clearance = 2.0; % Don't place it right against a wall
        rxPosition = [ ...
            modelBounds.xmin + rand * (modelBounds.xmax - modelBounds.xmin), ...
            modelBounds.ymin + rand * (modelBounds.ymax - modelBounds.ymin), ...
            1.5]; % Assume 1.5m height
        % Clamp position to ensure it's inside the cleared area
        rxPosition(1) = max(modelBounds.xmin + clearance, min(modelBounds.xmax - clearance, rxPosition(1)));
        rxPosition(2) = max(modelBounds.ymin + clearance, min(modelBounds.ymax - clearance, rxPosition(2)));
        fprintf('Random receiver placed at: [%.2f, %.2f, %.2f] m\n', rxPosition);
    end
    
    % Give the receiver a unique name for visualization
    if positionSweepEnabled
        rxSiteName = sprintf('UE_Actual_P%d_T%d', sweepIdx, trialIdx);
    else
        rxSiteName = 'UE_Actual';
    end
    
    % Create the Communications Toolbox 'rxsite' object
    rxSite = rxsite('cartesian','AntennaPosition',rxPosition(:),'Name',rxSiteName);
    fprintf('Receiver site created successfully.\n');
catch ME
    fprintf('Error in receiver placement: %s\n', ME.message);
    return;
end

%% Step 3: Transmitter Placement
fprintf('\n=== Step 3: Transmitter Placement ===\n');
try
    if config.enableStaticPositions
        % Use the hardcoded [N x 3] matrix of positions from the config
        fprintf('Using hardcoded transmitter positions from configuration.\n');
        txPositions = config.staticTxPositions;
        config.numTx = size(txPositions,1); % Update numTx to match the list
        fprintf('Set number of transmitters to %d based on static list.\n', config.numTx);
    else
        % Use the dynamic placement algorithm
        fprintf('Using enhanced dynamic transmitter placement...\n');
        % This function calculates optimal Tx positions based on room bounds,
        % desired number of Txs, and the *current* Rx position.
        txPositions = enhancedTransmitterPlacement(modelBounds, config.numTx, rxPosition, config.heightConstraints);
    end
catch ME
    fprintf('Error in transmitter placement: %s\n', ME.message);
    return;
end

%% Step 4: Transmitter Sites, Carrier and PRS Configuration
fprintf('\n=== Step 4: Transmitter Site Creation and Configuration ===\n');
try
    numTx = config.numTx;
    txSites = cell(numTx, 1);    % Cell array for txsite objects
    carriers = cell(numTx, 1);   % Cell array for nrCarrierConfig objects
    prsConfigs = cell(numTx, 1); % Cell array for nrPRSConfig objects
    
    % --- Configure 5G PRS signals ---
    % Goal: Give each Tx a unique signal so they can be differentiated.
    combSizes = config.prsConfig.combSizeOptions; % e.g., [2, 4, 6]
    try
        % Assign a unique NPRSID (0-4095) to each transmitter
        uniquePRSID = randperm(config.prsConfig.maxPRSID, numTx) - 1;
    catch
        % Fallback if randperm fails
        uniquePRSID = 0:numTx-1;
    end
    
    % Check if we can use 'parfor' (parallel for loop)
    useParallelTxConfig = parallelFlags.waveform && numTx > 1;
    if useParallelTxConfig
        % Create all transmitter configs in parallel for speed
        parfor i = 1:numTx
            [txSites{i}, carriers{i}, prsConfigs{i}] = createTransmitterConfig(i, txPositions, config, uniquePRSID, combSizes);
        end
    else
        % Create all transmitter configs sequentially
        for i = 1:numTx
            [txSites{i}, carriers{i}, prsConfigs{i}] = createTransmitterConfig(i, txPositions, config, uniquePRSID, combSizes);
        end
    end
    fprintf('Successfully created %d transmitter sites and configurations.\n', numTx);
catch ME
    fprintf('Error in transmitter site creation: %s\n', ME.message);
    return;
end

%% Step 5: Site Visualization
fprintf('\n=== Step 5: Site Visualization ===\n');
try
    if isvalid(viewer)
        % Show the 'rxSite' (UE) on the 3D map
        show(rxSite, 'Map', viewer, 'ShowAntennaHeight', false, 'IconSize', [15,15]);
        % Show all 'txSites' (gNBs) on the 3D map
        for i = 1:numTx
            show(txSites{i}, 'Map', viewer, 'ShowAntennaHeight', false, 'IconSize', [15,15]);
        end
        fprintf('Sites visualized successfully.\n');
    else
        fprintf('Viewer handle is invalid. Skipping site visualization.\n');
    end
catch ME
     fprintf('Error in site visualization: %s\n', ME.message);
end

%% Step 7: Ray Tracing and Propagation
% This is one of the most computationally expensive steps.
% It calculates the exact paths (including reflections) from each Tx to the Rx.
fprintf('\n=== Step 7: Ray Tracing and Propagation ===\n');
try
    rays             = cell(numTx,1);   % To store the ray objects
    propagationDelay = inf(numTx,1); % To store the shortest path delay (sec)
    propagationLoss  = inf(numTx,1); % To store the shortest path loss (dB)
    available        = false(numTx,1);% Boolean flag: Can this Tx be heard?
    isLOS            = false(numTx,1);% Boolean flag: Is it Line-of-Sight?

    % --- Caching Setup ---
    % Ray tracing is VERY slow. This caching logic saves the results
    % to a file. If we run the *exact* same scenario again (same map,
    % same Tx/Rx positions), it will load the result from disk instead
    % of re-calculating it.
    cacheEnabled       = false;
    cacheDir           = '';
    cacheEnvSignature  = []; % A "hash" of the environment
    if isprop(config,'enableRaytraceCache') && config.enableRaytraceCache
        cacheEnabled = true;
        cacheDir = fullfile(pwd, 'raytrace_cache'); % Cache sub-folder
        if ~exist(cacheDir, 'dir')
            mkdir(cacheDir); % Create it if it doesn't exist
        end
        try
            % Create a unique signature for the map and its settings
            cacheEnvSignature = buildRaytraceEnvironmentSignature(config, modelBounds, roomDims);
        catch ME_cacheSig
            cacheEnabled = false;
            if config.enableDebug
                fprintf('Warning: Failed to build ray-trace cache signature (%s). Disabling cache.\n', ME_cacheSig.message);
            end
        end
    end

    % Loop through each transmitter to calculate its path to the receiver
    for i = 1:numTx
        txPosition = txSites{i}.AntennaPosition';
        distance   = norm(txPosition - rxPosition); % Straight-line distance
        fprintf('Processing Tx%d at distance %.1fm...\n', i, distance);

        cacheHit        = false;
        cacheKey        = '';
        cacheSignature  = [];

        if cacheEnabled
            try
                % Create a unique key for *this specific Tx-Rx link*
                [cacheKey, cacheSignature] = computeRaytraceCacheKey(cacheEnvSignature, txPosition, rxPosition, config.txFreq, config.rayTracingConfig);
                % Check if a file with this key already exists
                [cacheHit, cachedRays] = tryLoadRaytraceCache(cacheDir, cacheKey);
                if cacheHit
                    rays{i} = cachedRays; % Load from disk
                    fprintf('Tx%d: Loaded ray tracing result from cache.\n', i);
                end
            catch ME_cacheLookup
                cacheHit = false;
                if config.enableDebug
                    fprintf('Tx%d: Ray-trace cache lookup failed (%s). Continuing without cache.\n', i, ME_cacheLookup.message);
                end
            end
        end

        % --- Run Ray Tracing (if cache missed) ---
        if ~cacheHit
            try
                % Create a ray tracing propagation model
                pm = propagationModel("raytracing", ...
                    "CoordinateSystem","cartesian", ...
                    "MaxNumReflections", config.rayTracingConfig.MaxNumReflections, ...
                    "MaxNumDiffractions",config.rayTracingConfig.MaxNumDiffractions, ...
                    "SurfaceMaterial",   config.rayTracingConfig.SurfaceMaterial, ...
                    "UseGPU",            "on"); % Use GPU if available
                
                % This is the main ray tracing command
                rays{i} = raytrace(txSites{i}, rxSite, pm, "Map", viewer);

                % Save the result to the cache for next time
                if cacheEnabled && ~isempty(rays{i}) && ~isempty(cacheKey)
                    try
                        saveRaytraceCache(cacheDir, cacheKey, rays{i}, cacheSignature);
                    catch ME_cacheSave
                        if config.enableDebug
                            fprintf('Tx%d: Skipped caching of ray-trace result (%s).\n', i, ME_cacheSave.message);
                        end
                    end
                end
            catch ME_raytrace
                rays{i} = []; % Ray tracing failed
                fprintf('Tx%d: Ray tracing failed (%s). Marking unavailable.\n', i, ME_raytrace.message);
            end
        end

        % --- Process Ray Results ---
        rchk = rays{i};
        if iscell(rchk), rchk = rchk{1}; end % Handle cell array output

        if ~isempty(rchk) % If any rays were found
            try
                % Get the propagation delay for all found paths
                delays = arrayfun(@(r) r.PropagationDelay, rchk);
                % Find the shortest delay (the first-arriving path)
                [propagationDelay(i), idxMin] = min(delays);
            catch
                % Fallback: use straight-line distance
                propagationDelay(i) = distance/physconst('LightSpeed');
                idxMin = 1;
            end
            
            try
                % Get the path loss for that first-arriving path
                pl = rchk(idxMin).PathLoss;
                if ~isempty(pl)
                    propagationLoss(i) = pl;
                end
            catch
                % Fallback: use a basic path loss model
                propagationLoss(i) = calculateBasicPathLoss(distance, config);
            end
            
            % Mark this transmitter as usable
            available(i) = true;
            % Record if it was Line-of-Sight
            isLOS(i) = (rchk(idxMin).LineOfSight);

            if isfinite(propagationLoss(i))
                fprintf('Tx%d: Ray tracing - Delay=%.2fns, Loss=%.1fdB, LOS=%d\n', i, 1e9*propagationDelay(i), propagationLoss(i), isLOS(i));
            else
                fprintf('Tx%d: Ray tracing - Delay=%.2fns\n', i, 1e9*propagationDelay(i));
            end
        else
            % No rays found (e.g., blocked by concrete)
            fprintf('Tx%d: No valid rays (blocked). Marking unavailable.\n', i);
        end
    end
    fprintf('Propagation modeling completed. Available transmitters: %d/%d\n', nnz(available), numTx);
catch ME
    fprintf('Error in ray tracing: %s\n', ME.message);
    return;
end

%% Step 8: Waveform Generation and Channel Modeling
% Goal: 1. Create the ideal 5G PRS signal for each Tx.
%       2. Pass that signal through the ray-traced channel.
%       3. Sum all received signals at the UE.
%       4. Add noise (SNR).
fprintf('\n=== Step 8: Waveform Generation ===\n');
try
    % Get the sample rate from the first transmitter's config
    sampleRate   = nrOFDMInfo(carriers{1}).SampleRate;
    refWaveforms = cell(numTx,1); % To store ideal Tx waveforms
    refGrids     = cell(numTx,1); % To store ideal Tx resource grids

    % Option to use 'single' precision to save RAM
    useSingleBuffers = cfgGet(config,'useSinglePrecisionBuffers',false);
    waveformParallel = parallelFlags.waveform && numTx > 1;
    
    if waveformParallel
        % --- Generate Waveforms in Parallel ---
        waveformLengths = zeros(numTx,1);
        parfor i = 1:numTx
            % Call helper to generate the 5G PRS waveform
            [wf, gridLocal] = generateEnhancedPRSWaveform(carriers{i}, prsConfigs{i}, config.numSlots);
            if useSingleBuffers
                refWaveforms{i} = single(wf);
                refGrids{i} = single(gridLocal);
            else
                refWaveforms{i} = wf;
                refGrids{i} = gridLocal;
            end
            waveformLengths(i) = size(wf,1);
        end
        for i = 1:numTx
            if useSingleBuffers
                fprintf('  Cell %d waveform: %dx1 samples (single)\n', i, waveformLengths(i));
            else
                fprintf('  Cell %d waveform: %dx1 samples\n', i, waveformLengths(i));
            end
        end
    else
        % --- Generate Waveforms Sequentially ---
        for i = 1:numTx
            [wf, gridLocal] = generateEnhancedPRSWaveform(carriers{i}, prsConfigs{i}, config.numSlots);
            if useSingleBuffers
                refWaveforms{i} = single(wf);
                refGrids{i} = single(gridLocal);
            else
                refWaveforms{i} = wf;
                refGrids{i} = gridLocal;
            end
            if useSingleBuffers
                fprintf('  Cell %d waveform: %dx1 samples (single)\n', i, size(wf,1));
            else
                fprintf('  Cell %d waveform: %dx1 samples\n', i, size(wf,1));
            end
        end
    end

    fprintf('Using channel model: comm.RayTracingChannel\n');
    totalReceivedSignal = []; % This will be the final signal at the UE
    maxSigLen = 0;

    % --- Channel Modeling (Parallel) ---
    % This section simulates passing each waveform through its
    % corresponding ray-traced channel.
    channelParallel = parallelFlags.channel && nnz(available) > 1;
    poolWorkers = parallelFlags.poolSize;
    if channelParallel
        % Cap concurrency to avoid running out of memory
        maxConcurrent = min([parallelFlags.maxChannelWorkers, poolWorkers, nnz(available)]);
        if maxConcurrent < 2
            channelParallel = false; % Not enough workers
        else
            fprintf('  Channel processing concurrency capped at %d worker(s).\n', maxConcurrent);
        end
    end

    if channelParallel
        % --- Use 'parfeval' for Asynchronous Parallel Channel Sim ---
        % 'parfeval' is better than 'parfor' here because some channels
        % (with many reflections) are much slower to simulate than others.
        % This acts like a job queue.
        futures = parallel.FevalFuture.empty(0,1); % Array to hold job futures
        for i = 1:numTx
            if ~available(i)
                fprintf('  Tx%d unavailable after ray tracing. Skipping in channel modeling.\n', i);
                continue;
            end
            % If the job queue is full, wait for one to finish
            if numel(futures) >= maxConcurrent
                % 'fetchNext' waits for the *next* job to finish, not
                % necessarily the first one submitted.
                [fIdx, idxDone, y, errMsg] = fetchNext(futures);
                futures(fIdx) = []; % Remove completed job from queue
                
                % ... (Error handling for OOM in worker) ...
                
                if ~isempty(errMsg)
                    available(idxDone) = false; % Mark Tx as failed
                    fprintf('  Tx%d channel modeling failed: %s. Skipping.\n', idxDone, retryMsg);
                else
                    % Add the received signal to the total
                    [totalReceivedSignal, maxSigLen] = accumulateSignal(totalReceivedSignal, y);
                end
            end
            % Submit a new job to the queue
            % This calls 'runRayTracingChannelAsync' (helper fn) on a worker
            futures(end+1) = parfeval(@runRayTracingChannelAsync, 3, i, rays{i}, txSites{i}, rxSite, sampleRate, refWaveforms{i});
        end
        % --- Collect remaining jobs ---
        while ~isempty(futures)
            [fIdx, idxDone, y, errMsg] = fetchNext(futures);
            futures(fIdx) = []; % Remove from queue
            
            % ... (Error handling for OOM in worker) ...
            
            if ~isempty(errMsg)
                available(idxDone) = false;
                fprintf('  Tx%d channel modeling failed: %s. Skipping.\n', idxDone, retryMsg);
            else
                % Add the received signal to the total
                [totalReceivedSignal, maxSigLen] = accumulateSignal(totalReceivedSignal, y);
            end
        end
    else
        % --- Run Channel Modeling Sequentially ---
        for i = 1:numTx
            if ~available(i)
                fprintf('  Tx%d unavailable after ray tracing. Skipping in channel modeling.\n', i);
                continue;
            end
            % Call the helper function directly
            [idxDone, y, errMsg] = runRayTracingChannelAsync(i, rays{i}, txSites{i}, rxSite, sampleRate, refWaveforms{i});
            if ~isempty(errMsg)
                available(idxDone) = false;
                fprintf('  Tx%d channel modeling failed: %s. Skipping.\n', idxDone, errMsg);
                continue;
            end
            % Add the received signal to the total
            [totalReceivedSignal, maxSigLen] = accumulateSignal(totalReceivedSignal, y);
        end
    end

    if isempty(totalReceivedSignal)
        error('No signal received from any transmitter.');
    end

    % Convert to double precision (required for 'awgn' and estimators)
    totalReceivedSignal = double(totalReceivedSignal);

    % --- Add AWGN (Noise) ---
    % This is where the configured 'SNR_dB' is finally applied.
    try
        % Use Communications Toolbox 'awgn' function
        snrDb = config.channelConfig.SNR_dB;
        totalReceivedSignal = awgn(totalReceivedSignal, snrDb, 'measured');
    catch
        % Fallback: manual complex AWGN
        snrDb = config.channelConfig.SNR_dB;
        sigPow = mean(abs(totalReceivedSignal).^2);
        snrLin = 10^(snrDb/10);
        nVar   = max(sigPow/snrLin, eps);
        n = sqrt(nVar/2) * (randn(size(totalReceivedSignal)) + 1j*randn(size(totalReceivedSignal)));
        totalReceivedSignal = totalReceivedSignal + n;
    end
    fprintf('Channel modeling complete; AWGN added at %.1f dB SNR.\n', snrDb);

catch ME
    fprintf('Error in waveform generation or channel modeling: %s\n', ME.message);
    return;
end


%% Step 9: TDOA Estimation
% Goal: Correlate the 'totalReceivedSignal' against each transmitter's
% ideal 'refGrid' to find the Time of Arrival (TOA) for each. Then,
% convert these TOAs into Time Difference of Arrival (TDOA).
fprintf('\n=== Step 9: TDOA Estimation ===\n');
try
    % Get detection threshold from config
    detThr = 0.10; 
    try, detThr = config.tdoaConfig.correlationThreshold; catch, end

    % Call the helper function that runs 'nrTimingEstimate' for all Txs
    % 'estimatedDelays' = TDOA in seconds (relative to refIdx)
    % 'maxCorr'         = Normalized correlation strength [0, 1]
    % 'refIdx'          = The index of the Tx used as the reference
    [estimatedDelays, maxCorr, refIdx] = tdoaUsingNrTimingEstimate( ...
        totalReceivedSignal, refGrids, sampleRate, carriers, detThr, parallelFlags);
    
    % --- Convert TDOA (time) to Range Difference (distance) ---
    % This is the key input for the geometric position solver.
    tdoaMeasurements = physconst('LightSpeed') * estimatedDelays;

    fprintf('Reference transmitter: Tx%d\n', refIdx);
    fprintf('TDOA measurements (relative to Tx%d):\n', refIdx);
    txReportOrder = 1:numTx;
    for idxRep = txReportOrder
        if idxRep ~= refIdx && isfinite(tdoaMeasurements(idxRep))
            % Print the range difference (e.g., "Tx2 is 3.4m further away
            % from the UE than the reference Tx10")
            fprintf('  Tx%d - Tx%d: %.3f m (corr_norm: %.3f)\n', idxRep, refIdx, tdoaMeasurements(idxRep), maxCorr(idxRep));
        end
    end

    % Optional debug print
    if isprop(config,'enableDebug') && config.enableDebug
        topK = min(5, numTx);
        [sortedCorr, idx] = sort(maxCorr,'descend');
        fprintf('Top-%d anchors by metric: %s\n', topK, mat2str([idx(1:topK)'; sortedCorr(1:topK)']));
    end
catch ME
    fprintf('Error in TDOA estimation: %s\n', ME.message);
    return;
end

%% Step 10: Position Estimation
% Goal: Use the 'tdoaMeasurements' (range differences) to solve the
% system of hyperbolic equations for the receiver's [x, y, z] position.
%
% This is the most complex logic block, involving:
% 1. Anchor Selection (choosing the best Txs)
% 2. RANSAC (to reject outlier measurements)
% 3. Coarse Initialization (to get a "good guess")
% 4. Non-Linear Solve (Gauss-Newton or lsqnonlin)
% 5. Fallback logic (in case the solve fails)
%
fprintf('\n=== Step 10: Position Estimation ===\n');
force2D = false;        % Flag for 2D-only solve
zFixed = NaN;           % The fixed height to use for 2D solves
solverTag = '';         % For logging which solver was used
validTx = [];           % Final set of non-ref anchors used
baselineTx = [];        % Initial set for coarse solve
cand = [];              % All anchors that passed threshold
ransacInlierMask = [];  % Boolean mask from RANSAC
gdopEstimate = NaN;     % GDOP metric
gateMeters = 30;        % Gating/threshold for RANSAC/residuals
posCoarse = [];         % The "initial guess" for the solver
coarseCov = [];         % Covariance from coarse solve
residualVector = NaN(numTx,1); % Final TDOA error (m) for each Tx
centroidFallback = false;      % Flag: True if we had to use the fallback
centroidReason = '';           % Reason for falling back
geometryPoor = false;          % Flag: True if GDOP is bad
try
    fprintf('Estimating position using hyperbolic multilateration...\n');

    % Check if we have at least 3 valid TDOA measurements (e.g., Ref + 2 others)
    if sum(~isnan(estimatedDelays)) >= 3

        % --- Configure Solver Dimensionality (2D vs 3D) ---
        try
            if isprop(config,'force2DPositioning')
                force2D = logical(config.force2DPositioning);
            end
        catch
        end

        % Get the fixed Z-height to use if in 2D mode
        zFixed = NaN;
        try
            if isprop(config,'defaultUEHeight')
                defaultHeight = config.defaultUEHeight;
                if isfinite(defaultHeight), zFixed = defaultHeight; end
            end
            if ~isfinite(zFixed) && isprop(config,'heightConstraints') && isfield(config.heightConstraints,'preferredHeight')
                preferredHeight = config.heightConstraints.preferredHeight;
                if isfinite(preferredHeight), zFixed = preferredHeight; end
            end
        catch
        end
        if ~isfinite(zFixed), zFixed = 1.5; end % Default to 1.5m

        centroidFallback = false;
        centroidReason = '';
        geometryPoor = false;

        try
            % --- 1. Initial Anchor Selection ---
            
            % Get all non-reference anchors that passed the detection
            % threshold and have a valid (non-NaN) TDOA measurement.
            cand = find(maxCorr > detThr & isfinite(tdoaMeasurements));
            cand = cand(cand ~= refIdx); % Remove reference Tx

            % Determine minimum anchors required by the solver
            requiredAnchors = 3 + (~force2D); % 3 for 2D, 4 for 3D

            % If we don't have enough, relax the threshold and take the
            % top-K strongest anchors just to have *something* to work with.
            if numel(cand) < requiredAnchors
                [~, globalOrder] = sort(maxCorr, 'descend');
                globalOrder = globalOrder(isfinite(tdoaMeasurements(globalOrder)));
                globalOrder = globalOrder(globalOrder ~= refIdx);
                if isempty(globalOrder)
                    error('No anchors have finite TDOA measurements.');
                end
                relaxCount = min(10, numel(globalOrder));
                extra = globalOrder(1:relaxCount);
                cand = unique([cand(:); extra(:)], 'stable');
                fprintf('Relaxed detection threshold to retain top-%d anchors (min set for solver).\n', numel(cand));
            end
            if isempty(cand)
                error('No anchors passed the detection threshold.');
            end

            % Rank the candidates by correlation strength
            [~, ordCorr] = sort(maxCorr(cand), 'descend');

            % Cap the number of anchors to a max (e.g., 10)
            maxAnchors = 10; % default ceiling
            try
                if isprop(config,'tdoaConfig') && isfield(config.tdoaConfig,'bestK') && isfinite(config.tdoaConfig.bestK)
                    maxAnchors = max(requiredAnchors, config.tdoaConfig.bestK);
                else
                    maxAnchors = max(requiredAnchors, maxAnchors);
                end
            catch
                maxAnchors = max(requiredAnchors, maxAnchors);
            end

            % Take the best N anchors
            Ncand  = min(maxAnchors, numel(ordCorr));
            validTx = cand(ordCorr(1:Ncand));
            validTx = validTx(validTx ~= refIdx); % Ensure ref is not in list
            
            % ... (logic to ensure minimum anchor count) ...

            % --- 2. Anchor Geometry Check (GDOP) ---
            
            % Prune the list to a set with good *angular diversity*
            % This prevents using 5 anchors that are all in a straight line.
            desiredCount = min(maxAnchors, numel(validTx));
            desiredCount = max(requiredAnchors, desiredCount);
            validTx = selectAnchorsWithAngularSpread(txPositions, refIdx, validTx, maxCorr, desiredCount);

            % Estimate the 2D GDOP of this anchor set
            gdopLimit = cfgGet(config.tdoaConfig,'maxGDOP',10);
            gdopEstimate = estimate2DTDOAGDOP(txPositions, refIdx, validTx);
            
            % **FALLBACK 1: POOR GEOMETRY**
            if ~isfinite(gdopEstimate) || gdopEstimate > gdopLimit
                % If GDOP is bad, set a flag to use the centroid fallback.
                % The geometry is too unstable for a hyperbolic solve.
                geometryPoor = true;
                centroidReason = sprintf('GDOP %.2f exceeds limit %.2f', gdopEstimate, gdopLimit);
                fprintf('Anchor geometry GDOP %.2f exceeds limit %.2f. Triggering centroid fallback.\n', gdopEstimate, gdopLimit);
            else
                fprintf('Anchor geometry GDOP estimate: %.2f (limit %.2f)\n', gdopEstimate, gdopLimit);
            end

            % --- 3. Coarse Initialization ---
            % The non-linear solvers (lsqnonlin, GN) need a "good first
            % guess" (posCoarse) to work well. We get this guess using a
            % simpler, linear solver (WLLS or hyperbolic).
            
            measTDOA_m = tdoaMeasurements;
            posCoarse = [];
            coarseCov = [];
            gateMeters = 30; % Get TDOA gating value
            try
                if isprop(config,'tdoaConfig') && isfield(config.tdoaConfig,'expectedGateMeters') && isfinite(config.tdoaConfig.expectedGateMeters)
                    gateMeters = config.tdoaConfig.expectedGateMeters;
                end
            catch
            end
            
            % Select a baseline set (e.g., 4 best anchors) for the coarse solve
            baselineCount = min(max(requiredAnchors, 3), numel(validTx));
            baselineTx = validTx(1:baselineCount);

            if force2D
                % ... (Logic to find a 2D coarse position) ...
            else
                % ... (Logic to find a 3D coarse position using WLLS or GN) ...
            end
            
            % Last-resort coarse init: use the mean of the anchor positions
            if isempty(posCoarse) || any(~isfinite(posCoarse))
                posCoarse = mean(txPositions([refIdx; baselineTx],:),1).';
                if numel(posCoarse) == 2
                    posCoarse = [posCoarse(:); zFixed];
                end
            end

            % --- 4. RANSAC (RObust SAmple Consensus) ---
            % This is the key step to remove *outlier measurements* that
            % would otherwise corrupt the high-accuracy solver.
            fprintf('Using custom RANSAC to find robust set of inlier anchors...\n');
            try
                % Define RANSAC parameters
                ransacTrials = 100;         % How many random subsets to try
                ransacSampleSize = 3;       % Min non-ref anchors for a 2D solve
                ransacMaxDistance = 7.5;    % Inlier threshold in meters (TDOA residual)

                % Run the custom RANSAC helper function
                [model, inlierIndices] = ransacTDOA(txPositions, tdoaMeasurements, ...
                    refIdx, validTx, zFixed, ransacTrials, ransacMaxDistance, ...
                    ransacSampleSize, modelBounds);
                ransacInlierMask = inlierIndices; % Save mask for logging
            
                if ~isempty(model) && nnz(inlierIndices) >= requiredAnchors
                    % RANSAC found a consistent set of anchors
                    fprintf('RANSAC found a consistent model with %d inliers.\n', nnz(inlierIndices));
                    
                    % **CRITICAL STEP:** Update 'validTx' to *only* include
                    % the anchors that RANSAC agreed upon.
                    validTx = validTx(inlierIndices);
                    
                    % Use the RANSAC solution as the new, high-quality
                    % initial guess for the final solver.
                    posCoarse = model;
                else
                    fprintf('RANSAC failed to find a consistent model. Proceeding with original set of anchors.\n');
                end
            catch ME_ransac
                fprintf('Warning: Custom RANSAC algorithm failed: %s\n', ME_ransac.message);
            end

            % ... (Optional: refine coarse init again with the RANSAC inliers) ...
            
            % --- 5. Final Non-Linear Solve ---
            if geometryPoor
                % **FALLBACK 1 (from above):** Geometry was bad, use centroid.
                estimatedPosition = improvedCentroidPositioning(txPositions, maxCorr);
                centroidFallback = true;
            elseif length(validTx) >= requiredAnchors
                % We have enough *RANSAC-cleaned* anchors, proceed to solve.
                nNonRef = length(validTx);
                fprintf('Using %d non-reference anchors (plus ref Tx%d): total %d.\n', ...
                        nNonRef, refIdx, nNonRef+1);

                if force2D
                    % --- 2D Solve Path ---
                    fprintf('Forcing 2D positioning with fixed z=%.2f m\n', zFixed);
                    % ... (Call 2D solver, e.g., solveTDOA2D_GN) ...
                    
                else
                    % --- 3D Solve Path (Main) ---
                    canDo3D = (nNonRef >= 4);
                    hasLSQ  = (exist('lsqnonlin','file') == 2); % Check for Optimization Tbx
                    initialGuess = posCoarse;
                    
                    if canDo3D
                        % Call the robust 3D solver, passing in the
                        % RANSAC-cleaned 'validTx' list and the 'initialGuess'.
                        % This function internally tries 'lsqnonlin' or 'solveTDOA3D_GN'
                        % and prunes any remaining high-residual anchors.
                        [estimatedPosition, validTx, solverTag] = solveTDOA3D_Robust( ...
                            txPositions, tdoaMeasurements, refIdx, validTx, maxCorr, ...
                            hasLSQ, initialGuess, gateMeters);
                        if strcmp(solverTag,'lsqnonlin')
                            fprintf('3D nonlinear TDOA positioning (lsqnonlin) successful\n');
                        else
                            fprintf('3D Gauss-Newton TDOA positioning successful\n');
                        end
                        fprintf('Anchors retained after pruning: %d (plus ref Tx%d).\n', numel(validTx), refIdx);
                    else
                        % Not enough anchors for 3D, fall back to 2D
                        estimatedPosition = properHyperbolicPositioning(txPositions, tdoaMeasurements, refIdx, validTx, maxCorr, zFixed, modelBounds);
                    end
                end

                % --- 6. Post-Solve Residual Check ---
                if ~centroidFallback
                    try
                        % Check the final error (residual) of the solution
                        % against the measurements it used.
                        residualVec = computeTDOAResiduals(txPositions, tdoaMeasurements, refIdx, validTx, estimatedPosition);
                        % ... (store residuals for logging) ...
                        
                        % **FALLBACK 3: HIGH RESIDUAL**
                        if max(residualVec(isfinite(residualVec))) > max(5, gateMeters)
                            % If the "best" solution is still terrible
                            % (e.g., avg error > 5m), discard it.
                            fprintf('TDOA residual check failed. Falling back to centroid.\n');
                            estimatedPosition = improvedCentroidPositioning(txPositions, maxCorr);
                            centroidFallback = true;
                            centroidReason = 'Residual exceeds gate';
                        end
                    catch ME_residual
                        % ...
                    end
                end
            else
                error('Insufficient anchors for hyperbolic positioning.');
            end

        catch ME_hyperbolic
            % --- 7. FALLBACK 2: SOLVER CRASH ---
            % If *any* part of the 'try' block above failed (e.g.,
            % RANSAC found 0 inliers, lsqnonlin failed to converge),
            % we land here in the 'catch' block.
            fprintf('Hyperbolic positioning failed: %s\n', ME_hyperbolic.message);
            fprintf('Falling back to improved centroid positioning.\n');
            centroidFallback = true;
            if isempty(centroidReason)
                centroidReason = sprintf('Hyperbolic failure: %s', ME_hyperbolic.message);
            end
            % Use the simplest, most robust (but least accurate) method.
            estimatedPosition = improvedCentroidPositioning(txPositions, maxCorr);
        end

    else
        % --- 8. FALLBACK 4: INSUFFICIENT DETECTIONS ---
        % We didn't even have 3 TDOA measurements to start with.
        fprintf('Signal quality too low for positioning.\n');
        fprintf('Falling back to improved centroid positioning.\n');
        centroidFallback = true;
        centroidReason = 'Insufficient finite TDOA measurements';
        estimatedPosition = improvedCentroidPositioning(txPositions, maxCorr);
    end

    % --- 9. Final Sanity Check ---
    % One last check, even if the solver *succeeded*.
    if ~isempty(estimatedPosition) && all(isfinite(estimatedPosition))
        if estimatedPosition(1) < modelBounds.xmin || estimatedPosition(1) > modelBounds.xmax || ...
           estimatedPosition(2) < modelBounds.ymin || estimatedPosition(2) > modelBounds.ymax
            
            % **FALLBACK 5: OUT-OF-BOUNDS**
            % The solver converged, but to a position outside the room.
            % This is physically impossible, so discard it.
            fprintf('Warning: The hyperbolic solver produced an estimate outside the model bounds.\n');
            fprintf('         Discarding estimate [%.2f, %.2f, %.2f] and falling back to centroid.\n', estimatedPosition);
            
            estimatedPosition = improvedCentroidPositioning(txPositions, maxCorr);
            centroidFallback = true;
            if isempty(centroidReason)
                centroidReason = 'Solver produced out-of-bounds estimate';
            end
        end
    end

    % --- UKF Step 2 ---
    % If the UKF is enabled AND we got a high-quality geometric solution
    % (i.e., we did *not* use the centroid fallback)...
    if useUKF && ~centroidFallback
        % Create the 'measurement' vector [x; y] from the solver
        measurement = [estimatedPosition(1); estimatedPosition(2)];
        
        % Call the 'correct' method of the UKF object.
        % This is the core of the Kalman Filter. It intelligently blends
        % the filter's *prediction* (from Step 1.5) with the new *measurement*
        % to produce a new, smoothed, and more accurate state.
        correct(ukf, measurement);
        
        % The new estimated position is now the filtered state from the UKF
        % (which is stored in 'ukf.State').
        filteredPosition = [ukf.State(1); ukf.State(2); estimatedPosition(3)]; % Keep original Z
        fprintf('UKF filtered position: [%.2f, %.2f, %.2f] m\n', filteredPosition);
        
        % Overwrite the raw geometric solution with the UKF's filtered solution.
        % This is now the final position for this trial.
        estimatedPosition = filteredPosition;
    end

    % --- Calculate Final Error Metrics ---
    horizError = hypot(estimatedPosition(1) - rxPosition(1), ...
                       estimatedPosition(2) - rxPosition(2));
    vertError  = abs(estimatedPosition(3) - rxPosition(3));
    err3D      = norm(estimatedPosition(:) - rxPosition(:));
    
    fprintf('Position estimation: [%.2f, %.2f, %.2f] m\n', estimatedPosition);
    fprintf('Horizontal error: %.2f m; Vertical error: %.2f m; 3D error: %.2f m\n', ...
            horizError, vertError, err3D);
    
    % Use 2D Horizontal Error as the primary performance metric
    positionError = horizError;
    
    if centroidFallback
        if isempty(centroidReason)
            centroidReason = 'Centroid fallback invoked';
        end
        fprintf('Degraded positioning: centroid fallback used (%s).\n', centroidReason);
    end

catch ME
    % --- 10. FALLBACK 6: CATASTROPHIC FAILURE ---
    % A totally unexpected error occurred.
    fprintf('Error in position estimation: %s\n', ME.message);
    estimatedPosition = [0;0;1.5]; % Return a dummy position
    horizError = NaN;
    vertError = NaN;
    err3D = NaN;
    positionError = 999; % Use a high error value
end



%% Step 11: Visualization
fprintf('\n=== Step 11: Results Visualization ===\n');
try
    % Create a *new* 'txsite' object to represent the *estimated* position
    if positionSweepEnabled
        estSiteName = sprintf('UE_Estimated_P%d_T%d', sweepIdx, trialIdx);
    else
        estSiteName = 'UE_Estimated';
    end
    rxEstimated = txsite('cartesian', ...
        'AntennaPosition', estimatedPosition(:), ...
        'TransmitterFrequency', config.txFreq, ...
        'Name',estSiteName);
    
    % Show the estimated position on the 3D map (e.g., as a red 'X')
    show(rxEstimated,'Map',viewer,'ShowAntennaHeight',false,'IconSize',[20,20]);
    fprintf('Estimated position visualized.\n');
catch ME
    fprintf('Error in results visualization: %s\n', ME.message);
end

%% Step 12: Performance Analysis
fprintf('\n=== Step 12: Performance Analysis ===\n');
try
    % Call the helper function to print a detailed summary (GDOP,
    % path loss, coverage, etc.) to the command window.
    performanceAnalysis(txPositions, rxPosition, propagationLoss, maxCorr, estimatedPosition, ...
        positionError, roomDims, config);

    % Print a final, simple summary for this trial
    fprintf('\n=== Positioning Summary ===\n');
    fprintf('Actual UE position: [%.2f, %.2f, %.2f] m\n', rxPosition);
    fprintf('Estimated UE position: [%.2f, %.2f, %.2f] m\n', estimatedPosition);
    fprintf('3D positioning error: %.2f m\n', err3D);
    avgMetric = mean(maxCorr(isfinite(maxCorr) & maxCorr>0));
    fprintf('Average detection metric: %.4g\n', avgMetric);
    if positionError < 2.0
        fprintf('Performance: EXCELLENT (< 2m error)\n');
    elseif positionError < 5.0
        fprintf('Performance: GOOD (< 5m error)\n');
    elseif positionError < 10.0
        fprintf('Performance: ACCEPTABLE (< 10m error)\n');
    else
        fprintf('Performance: NEEDS OPTIMIZATION\n');
    end
catch ME
    fprintf('Error in performance analysis: %s\n', ME.message);
end

% --- Store All Results for This Trial ---
% Create a massive struct containing *every* piece of data from this
% single simulation run. This struct is stored in the 'sweepResults'
% cell array, which will be written to the CSV at the very end.
sweepResults{sweepIdx, trialIdx} = struct( ...
    'rxTrue', rxPosition(:).', ...
    'estimated', estimatedPosition(:).', ...
    'horizontalError', horizError, ...
    'verticalError', vertError, ...
    'error3D', err3D, ...
    'positionError', positionError, ...
    'maxCorr', maxCorr(:).', ...
    'tdoaSeconds', estimatedDelays(:).', ...
    'tdoaMeters', tdoaMeasurements(:).', ...
    'detectedMask', isfinite(estimatedDelays(:)).', ...
    'propagationLoss', propagationLoss(:).', ...
    'propagationDelay', propagationDelay(:).', ...
    'availableMask', available(:).', ...
    'isLOSMask', isLOS(:).', ...
    'txPositions', txPositions, ...
    'refTx', refIdx, ...
    'selectedAnchors', validTx(:).', ...
    'baselineAnchors', baselineTx(:).', ...
    'candidateAnchors', cand(:).', ...
    'ransacInlierMask', ransacInlierMask(:).', ...
    'gdopEstimate', gdopEstimate, ...
    'gateMeters', gateMeters, ...
    'force2D', force2D, ...
    'zFixed', zFixed, ...
    'geometryPoor', geometryPoor, ...
    'centroidFallback', centroidFallback, ...
    'centroidReason', string(centroidReason), ...
    'solverTag', string(solverTag), ...
    'residuals', residualVector(:).', ...
    'coarseEstimate', posCoarse(:).', ...
    'coarseCovariance', coarseCov, ...
    'snrDb', snrDb, ...
    'trialIndex', trialIdx, ...
    'positionIndex', sweepIdx, ...
    'scenarioTag', string(scenarioTag));

    end % --- End of Trial Loop (Monte Carlo) ---
end % --- End of Position Sweep Loop ---

% --- Post-Processing and Final Logging ---
% This code runs *after* all position sweeps and trials are complete.
if positionSweepEnabled && ~isempty(sweepPositions)
    fprintf('\n=== Position Sweep Summary ===\n');
    % --- Generate Summary Statistics & Plots ---
    resultMask = ~cellfun('isempty', sweepResults);
    resultsList = sweepResults(resultMask);
    if isempty(resultsList)
        fprintf('  No sweep results recorded.\n');
    else
        % Extract all horizontal errors from all runs
        numSamples = numel(resultsList);
        horizErrors = zeros(numSamples,1);
        pointIndexVec = zeros(numSamples,1);
        positionsMat = zeros(numSamples,3);
        maxCorrStrong = NaN(numSamples,1);
        for rsIdx = 1:numSamples
            r = resultsList{rsIdx};
            horizErrors(rsIdx) = r.positionError;
            pointIndexVec(rsIdx) = r.positionIndex;
            positionsMat(rsIdx,:) = r.rxTrue;
            corrVals = r.maxCorr;
            corrFinite = corrVals(isfinite(corrVals) & corrVals > 0);
            if ~isempty(corrFinite)
                maxCorrStrong(rsIdx) = max(corrFinite);
            end
        end

        % Calculate 95th Percentile Error (a key metric)
        sortedErr = sort(horizErrors);
        errMean = mean(horizErrors);
        errStd  = std(horizErrors);
        if isempty(sortedErr)
            err95 = NaN;
        else
            idx95 = max(1, ceil(0.95 * numel(sortedErr)));
            err95 = sortedErr(idx95);
        end
        fprintf('  Horizontal error stats (m): mean=%.2f, std=%.2f, 95%%=%.2f\n', errMean, errStd, err95);

        % --- Plotting ---
        try
            % Plot 1: Cumulative Distribution Function (CDF)
            figure('Name','Horizontal Error CDF');
            plot(sortedErr, (1:numel(sortedErr)).'/max(1,numel(sortedErr)), 'LineWidth', 1.5);
            grid on;
            xlabel('Horizontal error (m)');
            ylabel('CDF');
            title('Horizontal Error Empirical CDF');
        catch ME_plotCDF
            fprintf('  Warning: failed to generate CDF plot (%s)\n', ME_plotCDF.message);
        end

        try
            % Plot 2: Box plot of error at each position
            figure('Name','Horizontal Error per UE Position');
            try
                boxchart(pointIndexVec, horizErrors);
            catch
                boxplot(horizErrors, pointIndexVec);
            end
            grid on;
            xlabel('UE position index');
            ylabel('Horizontal error (m)');
            title('Horizontal Error Distribution per UE Position');
        catch ME_box
            fprintf('  Warning: failed to generate per-position box plot (%s)\n', ME_box.message);
        end

        try
            % Plot 3: Line plot of *mean* error along the path
            meanErrors = accumarray(pointIndexVec(:), horizErrors(:), [size(sweepPositions,1) 1], @mean, NaN);
            distAlong = sqrt(sum((sweepPositions - sweepPositions(1,:)).^2, 2));
            validIdx = ~isnan(meanErrors);
            figure('Name','Mean Horizontal Error Along Path');
            plot(distAlong(validIdx), meanErrors(validIdx), '-o', 'LineWidth', 1.5);
            grid on;
            xlabel('Distance along path (m)');
            ylabel('Mean horizontal error (m)');
            title('Mean Horizontal Error Along UE Path');
        catch ME_line
            fprintf('  Warning: failed to generate line plot (%s)\n', ME_line.message);
        end

        try
            % Plot 4: Scatter plot of error vs. signal quality
            if any(isfinite(maxCorrStrong))
                figure('Name','Horizontal Error vs PRS Metric');
                scatter(maxCorrStrong, horizErrors, 60, pointIndexVec, 'filled');
                grid on;
                xlabel('Strongest PRS correlation metric');
                ylabel('Horizontal error (m)');
                title('Horizontal Error vs. PRS Detection Quality');
                cb = colorbar;
                cb.Label.String = 'UE position index';
            else
                fprintf('  Skipping error-vs-metric scatter (no finite correlation metrics).\n');
            end
        catch ME_scatter
            fprintf('  Warning: failed to generate scatter plot (%s)\n', ME_scatter.message);
        end

        try
            % Plot 5: Top-down 2D heatmap of error
            figure('Name','Top-Down Horizontal Error Map');
            scatter(positionsMat(:,1), positionsMat(:,2), 120, horizErrors, 'filled');
            axis equal;
            grid on;
            xlabel('X (m)');
            ylabel('Y (m)');
            title('Top-Down Horizontal Error Map');
            cb = colorbar;
            cb.Label.String = 'Horizontal error (m)';
            colormap('hot');
        catch ME_heat
            fprintf('  Warning: failed to generate top-down error map (%s)\n', ME_heat.message);
        end
    end
end

% --- Final CSV Logging ---
if resultLoggingCfg.enable
    try
        % Call helper to convert the cell array of structs into a flat table
        resultsTable = flattenSweepResultsTable(sweepResults, scenarioTag, config.channelConfig.SNR_dB, useUKF);
        if ~isempty(resultsTable)
            % Determine write mode (overwrite or append)
            writeMode = 'overwrite';
            if appendResults || ~resultLoggingCfg.overwrite
                if isfile(resultLoggingCfg.outputFile)
                    writeMode = 'append';
                end
            end
            % Write the table to the CSV file
            writetable(resultsTable, resultLoggingCfg.outputFile, 'WriteMode', writeMode);
            fprintf('Result logging: wrote %d samples to %s (%s).\n', height(resultsTable), resultLoggingCfg.outputFile, writeMode);
        else
            fprintf('Result logging: sweep produced no samples; nothing written.\n');
        end
    catch ME_log
        fprintf('Warning: failed to log results to %s (%s).\n', resultLoggingCfg.outputFile, ME_log.message);
    end
end

fprintf('\n=== 5G Positioning Complete ===\n');
% Clean up the parallel pool if auto-shutdown is enabled
if ~isempty(parallelPoolObj) && parallelFlags.autoShutdown
    try
        delete(parallelPoolObj);
    catch ME_cleanup
        fprintf('Parallel pool shutdown warning: %s\n', ME_cleanup.message);
    end
end

end % --- End of function runPositioningScenario ---

%% ========================================================================
%% LOCAL HELPER FUNCTIONS
%% ========================================================================

function snapshot = captureConfigSnapshot(cfg)
%CAPTURECONFIGSNAPSHOT Return a struct snapshot of writable config props.
%   This is used to save the "clean" state of the config object before
%   each scenario run in a sweep.
    snapshot = struct();
    propNames = properties(cfg);
    for idx = 1:numel(propNames)
        propName = propNames{idx};
        try
            snapshot.(propName) = cfg.(propName);
        catch
            % Skip properties that cannot be read
        end
    end
end

function restoreConfigFromSnapshot(cfg, snapshot)
%RESTORECONFIGFROMSNAPSHOT Apply a previously stored configuration snapshot.
%   This resets the config object back to its original state.
    propNames = fieldnames(snapshot);
    for idx = 1:numel(propNames)
        propName = propNames{idx};
        try
            cfg.(propName) = snapshot.(propName);
        catch
            % Skip read-only or dependent properties
        end
    end
end

function scenarios = buildScenarioPlan(config, snrSweepCfg, scenarioTagEnv, snrOverrideRequested, snrOverrideVal, ukfOverrideRequested, ukfOverrideVal)
%BUILDSCENARIOPLAN Assemble the active SNR/UKF combinations for the sweep.
%   This function creates the "test matrix" of all scenarios to run.
    scenarios = struct('snr', {}, 'ukf', {}, 'tag', {});

    if snrSweepCfg.enable
        % --- Sweep Mode ---
        % Get the list of SNR values and UKF modes to test
        snrValues = snrSweepCfg.values(:).';
        if isempty(snrValues)
            snrValues = config.channelConfig.SNR_dB;
        end
        if isfield(snrSweepCfg, 'ukfModes')
            ukfModesRaw = snrSweepCfg.ukfModes;
        else
            ukfModesRaw = [];
        end
        if isempty(ukfModesRaw)
            ukfModesRaw = logical(config.enableUKFTracking);
        end
        ukfModes = logical(ukfModesRaw(:).');

        % Create the Cartesian product of all combinations
        scenarioIdx = 0;
        for snrVal = snrValues
            for ukfFlag = ukfModes
                scenarioIdx = scenarioIdx + 1;
                scenarios(scenarioIdx).snr = snrVal;
                scenarios(scenarioIdx).ukf = logical(ukfFlag);
                % Create a unique tag, e.g., "MyRun_SNR15_UKF1"
                if isfield(snrSweepCfg, 'tagPrefix') && ~isempty(snrSweepCfg.tagPrefix)
                    scenarios(scenarioIdx).tag = sprintf('%s_SNR%.1f_UKF%d', snrSweepCfg.tagPrefix, snrVal, ukfFlag);
                else
                    scenarios(scenarioIdx).tag = sprintf('SNR%.1f_UKF%d', snrVal, ukfFlag);
                end
            end
        end
    else
        % --- Single Run Mode ---
        % Create a single scenario using the (potentially overridden) config
        scenarios(1).snr = config.channelConfig.SNR_dB;
        scenarios(1).ukf = logical(config.enableUKFTracking);
        if snrOverrideRequested
            scenarios(1).snr = snrOverrideVal;
        end
        if ukfOverrideRequested
            scenarios(1).ukf = ukfOverrideVal ~= 0;
        end
        if isempty(scenarioTagEnv)
            scenarios(1).tag = 'default';
        else
            scenarios(1).tag = char(scenarioTagEnv);
        end
    end
end

function performMatlabSessionCleanup()
%PERFORMMATLABSESSIONCLEANUP Clear lingering MATLAB state before execution.
%   This is a more aggressive cleanup than 'clear all'.
    try
        evalin('base', 'clearvars -global');
    catch
    end
    try
        clear functions; % Clear persistent function variables
    catch
    end
    try
        clear classes; % Clear class definitions
    catch
    end
    try
        clear mex; % Unload MEX files
    catch
    end
    try
        % Shut down any existing parallel pool
        if exist('gcp','file')
            poolObj = gcp('nocreate');
            if ~isempty(poolObj)
                delete(poolObj);
            end
        end
    catch
    end
end

function resultsTable = flattenSweepResultsTable(sweepResults, scenarioTag, snrDb, ukfEnabled)
%FLATTENSWEEPRESULTSTABLE Convert per-trial structs into a results table.
%   This function unpacks the 'sweepResults' cell array of structs into a
%   single, flat MATLAB 'table' object that can be easily written to a CSV.
%   It also converts array data (like 'maxCorrSeries') into strings.

    mask = ~cellfun('isempty', sweepResults);
    resultsList = sweepResults(mask);
    if isempty(resultsList)
        resultsTable = table();
        return;
    end

    numSamples = numel(resultsList);
    
    % --- Pre-allocate columns ---
    scenarioStr = repmat(string(scenarioTag), numSamples, 1);
    configuredSnrCol = repmat(snrDb, numSamples, 1);
    ukfCol = repmat(logical(ukfEnabled), numSamples, 1);

    snrAppliedCol = configuredSnrCol;
    positionIdx = zeros(numSamples,1);
    trialIdx = zeros(numSamples,1);
    refTxIdx = NaN(numSamples,1);
    force2DCol = false(numSamples,1);
    zFixedCol = NaN(numSamples,1);
    gateMetersCol = NaN(numSamples,1);
    geometryPoorCol = false(numSamples,1);
    centroidFallbackCol = false(numSamples,1);
    centroidReasonStr = strings(numSamples,1);
    solverTagStr = strings(numSamples,1);

    rxTrueMat = NaN(numSamples,3);
    estMat = NaN(numSamples,3);
    horizErr = NaN(numSamples,1);
    vertErr = NaN(numSamples,1);
    err3D = NaN(numSamples,1);
    posErr = NaN(numSamples,1);

    corrStrong = NaN(numSamples,1);
    corrAvg = NaN(numSamples,1);
    corrDetected = zeros(numSamples,1);

    pathLossMean = NaN(numSamples,1);
    pathLossBest = NaN(numSamples,1);
    pathLossWorst = NaN(numSamples,1);
    availableTx = zeros(numSamples,1);

    % --- Columns for string-encoded arrays ---
    maxCorrSeries = strings(numSamples,1);
    tdoaMetersSeries = strings(numSamples,1);
    tdoaSecondsSeries = strings(numSamples,1);
    propLossSeries = strings(numSamples,1);
    propDelaySeries = strings(numSamples,1);
    availableMaskSeries = strings(numSamples,1);
    isLosMaskSeries = strings(numSamples,1);
    selectedAnchorsStr = strings(numSamples,1);
    baselineAnchorsStr = strings(numSamples,1);
    candidateAnchorsStr = strings(numSamples,1);
    ransacMaskStr = strings(numSamples,1);
    residualsStr = strings(numSamples,1);
    coarseEstimateStr = strings(numSamples,1);
    coarseCovStr = strings(numSamples,1);
    detectedMaskStr = strings(numSamples,1);
    txPositionsStr = strings(numSamples,1);

    % --- Loop through all results and populate columns ---
    for idx = 1:numSamples
        r = resultsList{idx}; % Get the struct for this trial

        % Populate scalar values
        if isfield(r,'scenarioTag'), scenarioStr(idx) = string(r.scenarioTag); end
        if isfield(r,'snrDb') && ~isempty(r.snrDb), snrAppliedCol(idx) = double(r.snrDb); end
        if isfield(r,'positionIndex'), positionIdx(idx) = double(r.positionIndex); end
        if isfield(r,'trialIndex'), trialIdx(idx) = double(r.trialIndex); end
        if isfield(r,'refTx'), refTxIdx(idx) = double(r.refTx); end
        if isfield(r,'force2D'), force2DCol(idx) = logical(r.force2D); end
        if isfield(r,'zFixed'), zFixedCol(idx) = double(r.zFixed); end
        if isfield(r,'gateMeters'), gateMetersCol(idx) = double(r.gateMeters); end
        if isfield(r,'geometryPoor'), geometryPoorCol(idx) = logical(r.geometryPoor); end
        if isfield(r,'centroidFallback'), centroidFallbackCol(idx) = logical(r.centroidFallback); end
        if isfield(r,'centroidReason'), centroidReasonStr(idx) = string(r.centroidReason); end
        if isfield(r,'solverTag'), solverTagStr(idx) = string(r.solverTag); end

        if isfield(r,'rxTrue'), rxTrueMat(idx,:) = double(r.rxTrue(:).'); end
        if isfield(r,'estimated'), estMat(idx,:) = double(r.estimated(:).'); end
        if isfield(r,'horizontalError'), horizErr(idx) = double(r.horizontalError); end
        if isfield(r,'verticalError'), vertErr(idx) = double(r.verticalError); end
        if isfield(r,'error3D'), err3D(idx) = double(r.error3D); end
        if isfield(r,'positionError'), posErr(idx) = double(r.positionError); end

        % Process array fields (calculate stats and stringify)
        if isfield(r,'maxCorr')
            corrVals = double(r.maxCorr(:));
            maxCorrSeries(idx) = numericArrayToString(corrVals); % [0.1 0.5 0.9] -> "[0.1;0.5;0.9]"
            finiteCorr = corrVals(isfinite(corrVals) & corrVals > 0);
            corrDetected(idx) = numel(finiteCorr);
            if ~isempty(finiteCorr)
                corrStrong(idx) = max(finiteCorr);
                corrAvg(idx) = mean(finiteCorr);
            end
        end

        if isfield(r,'tdoaMeters')
            tdoaMetersSeries(idx) = numericArrayToString(r.tdoaMeters);
        end
        if isfield(r,'tdoaSeconds')
            tdoaSecondsSeries(idx) = numericArrayToString(r.tdoaSeconds);
        end
        if isfield(r,'detectedMask')
            detectedMaskStr(idx) = numericArrayToString(r.detectedMask);
        end

        if isfield(r,'propagationLoss')
            plVals = double(r.propagationLoss(:));
            propLossSeries(idx) = numericArrayToString(plVals);
            finitePL = plVals(isfinite(plVals));
            availableTx(idx) = numel(finitePL);
            if ~isempty(finitePL)
                pathLossMean(idx) = mean(finitePL);
                pathLossBest(idx) = min(finitePL);
                pathLossWorst(idx) = max(finitePL);
            end
        end
        if isfield(r,'propagationDelay')
            propDelaySeries(idx) = numericArrayToString(r.propagationDelay);
        end
        if isfield(r,'availableMask')
            availableMaskSeries(idx) = numericArrayToString(r.availableMask);
        end
        if isfield(r,'isLOSMask')
            isLosMaskSeries(idx) = numericArrayToString(r.isLOSMask);
        end
        if isfield(r,'selectedAnchors')
            selectedAnchorsStr(idx) = numericArrayToString(r.selectedAnchors);
        end
        if isfield(r,'baselineAnchors')
            baselineAnchorsStr(idx) = numericArrayToString(r.baselineAnchors);
        end
        if isfield(r,'candidateAnchors')
            candidateAnchorsStr(idx) = numericArrayToString(r.candidateAnchors);
        end
        if isfield(r,'ransacInlierMask')
            ransacMaskStr(idx) = numericArrayToString(r.ransacInlierMask);
        end
        if isfield(r,'residuals')
            residualsStr(idx) = numericArrayToString(r.residuals);
        end
        if isfield(r,'coarseEstimate')
            coarseEstimateStr(idx) = numericArrayToString(r.coarseEstimate);
        end
        if isfield(r,'coarseCovariance')
            coarseCovStr(idx) = numericArrayToString(r.coarseCovariance);
        end
        if isfield(r,'txPositions')
            txPositionsStr(idx) = numericArrayToString(r.txPositions);
        end
    end

    % --- Create the final table ---
    resultsTable = table( ...
        scenarioStr, configuredSnrCol, snrAppliedCol, ukfCol, ...
        positionIdx, trialIdx, refTxIdx, force2DCol, zFixedCol, gateMetersCol, geometryPoorCol, ...
        centroidFallbackCol, centroidReasonStr, solverTagStr, ...
        rxTrueMat(:,1), rxTrueMat(:,2), rxTrueMat(:,3), ...
        estMat(:,1), estMat(:,2), estMat(:,3), ...
        horizErr, vertErr, err3D, posErr, ...
        corrStrong, corrAvg, corrDetected, ...
        pathLossMean, pathLossBest, pathLossWorst, availableTx, ...
        maxCorrSeries, tdoaMetersSeries, tdoaSecondsSeries, ...
        propLossSeries, propDelaySeries, availableMaskSeries, isLosMaskSeries, ...
        selectedAnchorsStr, baselineAnchorsStr, candidateAnchorsStr, ransacMaskStr, residualsStr, ...
        coarseEstimateStr, coarseCovStr, detectedMaskStr, txPositionsStr, ...
        'VariableNames', { ...
            'ScenarioTag','ConfiguredSNR_dB','AppliedSNR_dB','UKFEnabled', ...
            'PositionIndex','TrialIndex','RefTxIndex','Force2D','ZFixed','GateMeters','GeometryPoor', ...
            'CentroidFallback','CentroidReason','SolverTag', ...
            'TrueX','TrueY','TrueZ','EstX','EstY','EstZ', ...
            'HorizontalError','VerticalError','Error3D','PositionError', ...
            'MaxCorrStrongest','MaxCorrAverage','Detections', ...
            'PathLossMean','PathLossBest','PathLossWorst','AvailableTx', ...
            'MaxCorrSeries','TDOAMeters','TDOASeconds', ...
            'PropagationLossSeries','PropagationDelaySeries','AvailableMask','IsLOSMask', ...
            'SelectedAnchors','BaselineAnchors','CandidateAnchors','RansacInliers','ResidualSeries', ...
        'CoarseEstimate','CoarseCovariance','DetectedMask','TxPositions'});
end

function strOut = numericArrayToString(val)
%NUMERICARRAYTOSTRING Encode numeric or logical arrays as a compact string.
%   e.g., [1 2 3] becomes "[1 2 3]"
    if isstring(val) || ischar(val)
        strOut = string(val);
        return;
    end
    if isempty(val)
        strOut = "";
        return;
    end
    if isa(val,'logical')
        val = double(val);
    end
    if ~isnumeric(val)
        strOut = string(val);
        return;
    end
    % Use mat2str for a compact, MATLAB-readable string representation
    strOut = string(mat2str(val, 6));
end

function [txSite, carrier, prsCfg] = createTransmitterConfig(idx, txPositions, config, uniquePRSID, combSizes)
%CREATETRANSMITTERCONFIG Build txsite, carrier, and PRS objects for one anchor.
%   This is a helper function called by the 'parfor' loop in Step 4.
    
    % Create the txsite object (physical location)
    txSite = txsite('cartesian', ...
        'AntennaPosition', txPositions(idx,:)', ...
        'TransmitterFrequency', config.txFreq, ...
        'Name', sprintf('gNB_%d', idx));

    % Create the 5G NR carrier configuration
    carrier = nrCarrierConfig;
    carrier.SubcarrierSpacing = config.carrierConfig.SubcarrierSpacing;
    carrier.CyclicPrefix      = config.carrierConfig.CyclicPrefix;
    carrier.NSizeGrid         = config.carrierConfig.NSizeGrid;
    carrier.NStartGrid        = config.carrierConfig.NStartGrid;

    % Create the 5G NR Positioning Reference Signal (PRS) configuration
    prsCfg = nrPRSConfig;
    prsCfg.PRSResourceSetPeriod   = config.prsConfig.PRSResourceSetPeriod;
    prsCfg.NumRB                  = config.prsConfig.NumRB;
    prsCfg.RBOffset               = config.prsConfig.RBOffset;
    prsCfg.SymbolStart            = config.prsConfig.SymbolStart;
    prsCfg.NumPRSSymbols          = config.prsConfig.NumPRSSymbols;
    prsCfg.PRSResourceRepetition  = config.prsConfig.PRSResourceRepetition;
    prsCfg.PRSResourceTimeGap     = config.prsConfig.PRSResourceTimeGap;
    
    % --- Assign unique, non-interfering signal parameters ---
    prsCfg.NPRSID                 = uniquePRSID(idx); % Unique ID
    if isempty(combSizes)
        combValue = prsCfg.CombSize;
    else
        % Cycle through the comb sizes (e.g., 2, 4, 6, 2, 4, 6, ...)
        combValue = combSizes(mod(idx-1, numel(combSizes)) + 1);
    end
    prsCfg.CombSize = combValue; % Frequency domain interleaving
    prsCfg.REOffset = mod(idx-1, combValue); % Frequency domain offset
end

function [idxOut, y, errMsg] = runRayTracingChannelAsync(idx, rayDataIn, txSiteObj, rxSiteObj, sampleRate, waveform)
%RUNRAYTRACINGCHANNELASYNC Simulate the ray tracing channel for one anchor.
%   This is the function executed in parallel by 'parfeval' in Step 8.
%   It takes the 'rays' and the 'waveform' and simulates the channel.
    idxOut = idx;
    errMsg = '';
    y = []; % Output received waveform
    try
        % --- Input Handling ---
        rvec = rayDataIn;
        if iscell(rvec)
            if isempty(rvec), error('Empty ray cell container'); end
            rvec = rvec{1};
        end
        if isempty(rvec)
            error('Empty ray vector'); % No paths found
        end

        wfLocal = waveform;
        if isa(wfLocal,'single')
            wfLocal = double(wfLocal); % Ensure double precision
        end

        % --- Create the Channel Object ---
        % Use 'comm.RayTracingChannel' from Communications Toolbox,
        % feeding it the pre-calculated 'rvec' (rays).
        rt = comm.RayTracingChannel(rvec, txSiteObj, rxSiteObj);
        rt.SampleRate                = sampleRate;
        rt.MinimizePropagationDelay  = false;
        rt.NormalizeImpulseResponses = false;
        rt.NormalizeChannelOutputs   = false;

        % --- Run the Channel Simulation ---
        % This applies the delays, path losses, and phase shifts
        % from the rays to the input 'wfLocal'.
        y = rt(wfLocal);
        if size(y,2) > 1
            y = y(:,1);
        end

        % Remove filter delay artifacts
        try
            rtInfo = info(rt);
            filterDelay = rtInfo.ChannelFilterDelay;
            if filterDelay > 0 && filterDelay < length(y)
                y = y(filterDelay+1:end);
            end
        catch
        end
        release(rt); % Release the channel object
    catch ME
        % Catch errors (e.g., Out of Memory)
        errMsg = ME.message;
        y = [];
    end
    if ~isempty(y)
        y = double(y);
        if isrow(y)
            y = y.';
        end
    end
end

function [accumSig, maxLen] = accumulateSignal(accumSig, newSig)
%ACCUMULATESIGNAL Combine channel outputs and record the longest length.
%   This helper function sums waveforms of different lengths by
%   zero-padding the shorter one.
    if isempty(newSig)
        maxLen = length(accumSig);
        return;
    end
    if isrow(newSig)
        newSig = newSig.';
    end
    if isempty(accumSig)
        % This is the first signal
        accumSig = newSig;
        maxLen = length(accumSig);
        return;
    end
    
    % Pad the shorter signal with zeros to match the longer one
    lenNew = length(newSig);
    lenAcc = length(accumSig);
    if lenNew > lenAcc
        accumSig(end+1:lenNew) = 0;
    elseif lenAcc > lenNew
        newSig(end+1:lenAcc) = 0;
    end
    
    % Sum the signals (this simulates what the antenna receives)
    accumSig = accumSig + newSig;
    maxLen = length(accumSig);
end

function [poolObj, flags] = setupParallelProcessing(config)
%SETUPPARALLELPROCESSING Start parallel pools based on configuration.
%   This helper checks system memory, core count, and config flags
%   to safely initialize the Parallel Computing Toolbox pool.
    flags = struct('waveform', false, 'channel', false, 'autoShutdown', false, ...
        'maxChannelWorkers', 1, 'poolSize', 0, 'channelMemoryGuardGB', inf);
    poolObj = [];
    try
        parallelCfg = cfgGet(config, 'parallelProcessing', struct());
        if isempty(parallelCfg) || ~cfgGet(parallelCfg, 'enable', false)
            return; % Parallel processing is disabled in the config
        end
        if ~license('test','Distrib_Computing_Toolbox')
            fprintf('Parallel Computing Toolbox unavailable. Running sequentially.\n');
            return;
        end
        
        % ... (Complex logic to check available RAM and core counts) ...
        % ... (This logic determines the optimal number of 'workers') ...
        
        % For this code, let's assume the logic results in 'workers' > 1
        workers = 8; % Example
        
        flags.poolSize = workers;
        if workers < 2
            fprintf('Parallel processing request skipped (workers=%d).\n', workers);
            flags.waveform = false;
            flags.channel = false;
            return;
        end

        % Get or start the parallel pool
        poolObj = gcp('nocreate');
        if isempty(poolObj) || poolObj.NumWorkers ~= workers
            if ~isempty(poolObj)
                delete(poolObj); % Delete existing pool if size is wrong
            end
            poolObj = parpool('local', workers); % Start new pool
        end
        
        % Set the final flags based on config
        flags.poolSize = poolObj.NumWorkers;
        flags.waveform = cfgGet(parallelCfg, 'enableWaveformParallel', true) && flags.poolSize > 0;
        flags.channel = cfgGet(parallelCfg, 'enableChannelParallel', true) && flags.poolSize > 0;
        flags.autoShutdown = cfgGet(parallelCfg, 'autoShutdown', false);
        % ... (More logic to set memory/concurrency caps) ...
        
    catch ME
        fprintf('Parallel setup failed: %s\n', ME.message);
        % ... (reset all flags to false) ...
    end
end

function availableGB = tryGetAvailableMemoryGB()
%TRYGETAVAILABLEMEMORYGB Estimate available system memory in gigabytes.
    availableGB = NaN;
    try
        memInfo = memory; % Get memory information
        if isfield(memInfo, 'MemAvailableAllArrays')
            availableGB = double(memInfo.MemAvailableAllArrays) / 1024^3;
        elseif isfield(memInfo, 'MaxPossibleArrayBytes')
            availableGB = double(memInfo.MaxPossibleArrayBytes) / 1024^3;
        end
    catch
        availableGB = NaN;
    end
end

function envSignature = buildRaytraceEnvironmentSignature(config, modelBounds, roomDims)
%BUILDRAYTRACEENVIRONMENTSIGNATURE Produce a hashable description of the room.
%   This creates a struct that uniquely identifies the 3D environment.
%   Used by the caching system.
    envSignature = struct();
    envSignature.cacheVersion = 1;
    envSignature.useCustomModel = logical(config.useCustomModel);
    envSignature.customModelFile = char(config.customModelFile);
    envSignature.modelScaleFactor = double(config.modelScaleFactor);
    envSignature.roomDims = round(double(roomDims(:).'), 4);
    % ... (add bounds and material properties) ...
end

function [cacheKey, signaturePayload] = computeRaytraceCacheKey(envSignature, txPosition, rxPosition, txFreq, rtConfig)
%COMPUTERAYTRACECACHEKEY Generate identifiers for storing raytrace results.
%   This creates a unique signature for a *specific Tx-Rx link* within
%   the environment. It is then hashed (SHA-256) to create the 'cacheKey'.
    signaturePayload = struct();
    signaturePayload.version = envSignature.cacheVersion;
    signaturePayload.environment = envSignature;
    signaturePayload.txPosition = round(double(txPosition(:).'), 4);
    signaturePayload.rxPosition = round(double(rxPosition(:).'), 4);
    signaturePayload.txFrequency = double(txFreq);
    signaturePayload.rayTracingConfig = struct( ...
        'MaxNumReflections', rtConfig.MaxNumReflections, ...
        'MaxNumDiffractions', rtConfig.MaxNumDiffractions, ...
        'SurfaceMaterial', char(rtConfig.SurfaceMaterial), ...
        'CoordinateSystem', char(rtConfig.CoordinateSystem));

    % --- Hashing ---
    canonicalPayload = canonicalizeForRaytraceHash(signaturePayload);
    jsonText = jsonencode(canonicalPayload); % Convert struct to JSON string
    md = java.security.MessageDigest.getInstance('SHA-256'); % Get SHA-256 hash
    md.update(uint8(jsonText));
    hashBytes = typecast(uint8(md.digest()), 'uint8');
    cacheKey = lower(reshape(dec2hex(hashBytes)', 1, [])); % Convert hash to hex string
end

function dataOut = canonicalizeForRaytraceHash(dataIn)
%CANONICALIZEFORRAYTRACEHASH Normalize structures prior to hashing.
%   Ensures fields are in alphabetical order so hashing is consistent.
    if isstruct(dataIn)
        dataOut = orderfields(dataIn); % Sort struct fields
        fields = fieldnames(dataOut);
        for idx = 1:numel(fields)
            dataOut.(fields{idx}) = canonicalizeForRaytraceHash(dataOut.(fields{idx}));
        end
    % ... (handle other data types) ...
    else
        dataOut = dataIn;
    end
end

function [isHit, rayData] = tryLoadRaytraceCache(cacheDir, cacheKey)
%TRYLOADRAYTRACECACHE Retrieve cached ray-tracing results if available.
    cacheFile = fullfile(cacheDir, [cacheKey, '.mat']);
    isHit = false;
    rayData = [];
    if ~isfile(cacheFile)
        return; % Cache file doesn't exist
    end
    try
        % Load the .mat file
        S = load(cacheFile, 'rayData', 'cacheMeta');
        if isfield(S, 'cacheMeta') && isfield(S.cacheMeta, 'hash')
            % Final check: does the hash inside the file match the key?
            if strcmp(S.cacheMeta.hash, cacheKey)
                rayData = S.rayData; % Success!
                isHit = true;
            end
        end
    catch
        % File was corrupt
        isHit = false;
        rayData = [];
    end
end

function saveRaytraceCache(cacheDir, cacheKey, rayData, signaturePayload)
%SAVERAYTRACECACHE Persist ray-tracing results for later reuse.
    cacheFile = fullfile(cacheDir, [cacheKey, '.mat']);
    cacheMeta = struct();
    cacheMeta.hash = cacheKey;
    cacheMeta.signature = signaturePayload;
    cacheMeta.generatedOn = datestr(now, 30);
    % Save the data to a .mat file
    save(cacheFile, 'rayData', 'cacheMeta');
end

function [delayOut, corrOut, logMsg] = timingEstimateForTx(idx, carrierCfg, rxWaveform, refGrid, fs)
%TIMINGESTIMATEFORTX Run NR timing estimation for a single transmitter index.
%   This is the core of the TDOA estimation, called by 'parfor' in Step 9.
    delayOut = NaN;
    corrOut = 0;
    logMsg = '';
    rg = double(refGrid);
    if isempty(rg) || ~any(abs(rg(:)) > 0)
        % This Tx had no PRS signal in this slot (e.g., muting)
        logMsg = sprintf('  Tx%-2d: timing estimate skipped (empty PRS grid)\n', idx);
        return;
    end
    try
        % --- 5G Toolbox: Timing Estimation ---
        % This is the key function. It correlates the *entire* received
        % signal ('rxWaveform') against the *ideal* grid for this *one*
        % transmitter ('rg').
        [off, mag] = nrTimingEstimate(carrierCfg, rxWaveform, rg);
        
        if isempty(off) || ~isfinite(off(1))
            logMsg = sprintf('  Tx%-2d: timing estimate skipped (no finite offset)\n', idx);
            return;
        end

        % The offset (in samples) is the Time of Arrival (TOA)
        offsetInSamples = double(off(1));
        
        % ... (logic to find the peak magnitude) ...
        peakMag = double(max(mag(1:min(8192, numel(mag)))));

        if ~isfinite(peakMag)
            logMsg = sprintf('  Tx%-2d: timing estimate skipped (non-finite metric)\n', idx);
            return;
        end

        % Return the TOA in seconds and the correlation magnitude
        delayOut = offsetInSamples / fs;
        corrOut = peakMag;
        logMsg = sprintf('  Tx%-2d: TOA = %8.2f ns, metric=%.4g\n', idx, 1e9*delayOut, corrOut);
    catch ME
        logMsg = sprintf('  Tx%-2d: timing estimate failed: %s\n', idx, ME.message);
    end
end

function [estimatedDelays, maxCorr, refIdx] = tdoaUsingNrTimingEstimate( ...
    rxWaveform, refGrids, fs, carriers, corrThreshold, parallelFlags)
%TDOAUSINGNRTIMINGESTIMATE Compute per-anchor delays and correlations.
%   This function orchestrates Step 9. It calls 'timingEstimateForTx'
%   for all transmitters and then converts their absolute TOAs to
%   relative TDOAs.
    fprintf('Estimating TDOA using nrTimingEstimate (PRS) ...\n');

    numTx = numel(refGrids);
    estimatedDelays = NaN(numTx,1); % Will store TOA, then TDOA (sec)
    maxCorr         = zeros(numTx,1); % Will store correlation strength

    if isempty(rxWaveform) || size(rxWaveform,2) ~= 1
        error('rxWaveform must be a non-empty column vector.');
    end

    logEntries = repmat({''}, numTx, 1);
    % ... (Check for parallel) ...
    useParallelTDOA = false; % (Simplified for this comment)

    if useParallelTDOA
        % Run all TDOA estimations in parallel
        parfor i = 1:numTx
            [estimatedDelays(i), maxCorr(i), logEntries{i}] = timingEstimateForTx( ...
                i, carriers{i}, rxWaveform, refGrids{i}, fs);
        end
    else
        % Run all TDOA estimations sequentially
        for i = 1:numTx
            [estimatedDelays(i), maxCorr(i), logEntries{i}] = timingEstimateForTx( ...
                i, carriers{i}, rxWaveform, refGrids{i}, fs);
        end
    end

    % Print all log messages
    for i = 1:numTx
        if ~isempty(logEntries{i})
            fprintf('%s', logEntries{i});
        end
    end

    rawCorr = maxCorr;
    % Find all Txs that are valid (finite) and passed the threshold
    finiteMask = isfinite(estimatedDelays) & isfinite(rawCorr);
    passMask = finiteMask & (rawCorr >= corrThreshold);
    validIdx = find(passMask);

    if isempty(validIdx)
        % If *none* passed, relax the threshold and just take the
        % strongest K finite detections.
        fallbackIdx = find(finiteMask);
        if isempty(fallbackIdx)
            error('No valid PRS detections available for TDOA.');
        end
        [~, order] = sort(rawCorr(fallbackIdx), 'descend');
        fallbackCount = min(numel(order), max(1, 4));
        validIdx = fallbackIdx(order(1:fallbackCount));
        fprintf('Warning: PRS metrics below %.3f; using top-%d detections.\n', corrThreshold, numel(validIdx));
    end

    % Normalize correlation metric: scale the strongest valid Tx to 1.0
    maxCorr = zeros(size(rawCorr));
    scale = max(rawCorr(validIdx));
    if scale > 0
        maxCorr(validIdx) = rawCorr(validIdx) ./ scale;
    end
    % Discard any Txs that were finite but didn't pass the (relaxed) threshold
    invalidIdx = setdiff(find(finiteMask), validIdx);
    estimatedDelays(invalidIdx) = NaN;
    fprintf('PRS detections usable: %d/%d (relative metric normalized)\n', numel(validIdx), numTx);

    % --- Convert TOA to TDOA ---
    % 1. Select reference as the strongest PRS detection
    [~, k]  = max(maxCorr(validIdx));
    refIdx  = validIdx(k);

    % 2. Get the absolute TOA of the reference transmitter
    refDelay = estimatedDelays(refIdx);

    % 3. Subtract the reference delay from all other delays
    % This converts 'estimatedDelays' from TOA to TDOA.
    % e.g., [100ns, 120ns, 90ns] -> [10ns, 30ns, 0ns] (if refIdx=3)
    estimatedDelays = estimatedDelays - refDelay;
end

% ---------- TDOA Solver Helper Functions ----------

function pathLoss = calculateBasicPathLoss(distance, config)
%CALCULATEBASICPATHLOSS Fallback log-distance path-loss model.
    frequency = config.txFreq;
    fspl = 32.45 + 20*log10(max(distance,1e-3)) + 20*log10(frequency/1e9);
    indoorLoss = 20;
    pathLoss = fspl + indoorLoss + 5*randn; % Add some randomness
end

function [posCoarse, covCoarse] = coarseTDOA_WLLS(txPositions, tdoaMeters, refIdx, selectedTx, corrNorm)
%COARSETDOAWLLS Weighted least-squares TDOA initialization for 3D solving.
%   This is a linear (non-iterative) solver that provides a "good guess".
    posCoarse = [];
    covCoarse = [];
    if numel(selectedTx) < 3
        return;
    end
    try
        % ... (Implementation of WLLS algorithm) ...
    catch
        posCoarse = [];
        covCoarse = [];
    end
end

function estimatedPosition = properHyperbolicPositioning(txPositions, tdoaMeters, refIdx, selectedTx, corrNorm, fixedZ, modelBounds)
%PROPERHYPERBOLICPOSITIONING Solve TDOA using classical linear hyperbolic methods.
%   This is another (simpler) linear solver for getting a coarse position.
    try
        % ... (Implementation of linear hyperbolic solver) ...
    catch ME
        % ... (Fallback if solver fails) ...
    end
end

function estimatedPosition = improvedCentroidPositioning(txPositions, correlations)
%IMPROVEDCENTROIDPOSITIONING Derive centroid-based estimate weighted by correlation.
%   This is the *ultimate fallback* method. It is low-accuracy but
%   extremely robust. It calculates a weighted average of the positions
%   of the transmitters, where the weight is the correlation strength.
    fprintf('Using correlation-weighted centroid positioning\n');
    w = correlations.^2; % Square correlations to emphasize strong signals
    if ~any(w>0), w = ones(size(correlations)); end % Use 1s if all are 0
    w = w/sum(w); % Normalize weights
    if isrow(w), w=w'; end
    % Calculate weighted average of (x, y) positions
    xy = sum(txPositions(:,1:2).*w,1);
    % Return the (x, y) centroid at a default height
    estimatedPosition = [xy(1:2)'; 1.5];
end

function selectedTx = selectAnchorsWithAngularSpread(txPositions, refIdx, candidates, corrMetrics, desiredCount)
%SELECTANCHORSWITHANGULARSPREAD Choose anchors with sufficient angular diversity.
%   This function prevents selecting 5 anchors that are all in a straight
%   line, which would lead to poor GDOP and high position error. It tries
%   to pick anchors that "surround" the reference transmitter.
    if isempty(candidates)
        selectedTx = candidates;
        return;
    end
    % ... (Greedy algorithm to select anchors that maximize both
    %      correlation strength and angular separation) ...
    
    % (Simplified: for this, just assume it returns a subset)
    selectedTx = candidates(1:min(desiredCount, numel(candidates)));
    selectedTx = selectedTx(:);
end

function gdop = estimate2DTDOAGDOP(txPositions, refIdx, anchorSet)
%ESTIMATE2DTDOAGDOP Approximate GDOP for 2D TDOA anchor geometries.
%   Calculates the Geometric Dilution of Precision (GDOP) for the
%   selected anchor set. A low GDOP (<5) is good. A high GDOP (>10) is bad.
    gdop = inf;
    anchorSet = anchorSet(:)';
    anchorSet = anchorSet(anchorSet ~= refIdx);
    if numel(anchorSet) < 2
        return;
    end
    % ... (Matrix math to calculate GDOP) ...
    try
        points = txPositions([refIdx, anchorSet], 1:2);
        nominal = mean(points, 1);
        refVec = nominal - txPositions(refIdx,1:2);
        rr = norm(refVec);
        if rr < 1e-3
            return;
        end
        ur = refVec / rr;
        M = numel(anchorSet);
        H = zeros(M,2);
        for ii = 1:M
            idx = anchorSet(ii);
            uiVec = nominal - txPositions(idx,1:2);
            ri = norm(uiVec);
            if ri < 1e-3
                return;
            end
            ui = uiVec / ri;
            H(ii,:) = ui - ur;
        end
        G = H.' * H;
        if cond(G) > 1e8
            return;
        end
        gdop = sqrt(trace(inv(G)));
    catch
        gdop = inf;
    end
end

function [xhat, usedTx, solverTag] = solveTDOA3D_Robust(txPositions, tdoaMeters, refIdx, candidateTx, corrNorm, hasLSQ, initialGuess, gateMeters)
%SOLVETDOA3D_ROBUST Orchestrate robust 3D TDOA solving with fallbacks.
%   This function attempts to solve using 'lsqnonlin' (if available) or
%   'solveTDOA3D_GN'. It also iteratively removes high-residual anchors.
    anchorSet = candidateTx(:)';
    solverTag = 'gauss-newton';
    if nargin < 7 || isempty(initialGuess)
        initialGuess = [];
    end
    % ... (Logic for iterative residual-based pruning) ...
    
    % (Simplified logic for comment:)
    try
        if hasLSQ
            % Try Optimization Toolbox first
            xhat = solveTDOA3D_lsqnonlin(txPositions, tdoaMeters, refIdx, anchorSet, corrNorm, initialGuess);
            solverTag = 'lsqnonlin';
        else
            % Fallback to custom Gauss-Newton
            xhat = solveTDOA3D_GN(txPositions, tdoaMeters, refIdx, anchorSet, corrNorm, initialGuess);
            solverTag = 'gauss-newton';
        end
    catch
        % Fallback if lsqnonlin fails
        xhat = solveTDOA3D_GN(txPositions, tdoaMeters, refIdx, anchorSet, corrNorm, initialGuess);
        solverTag = 'gauss-newton';
    end
    usedTx = anchorSet;
end

function residuals = computeTDOAResiduals(txPositions, tdoaMeters, refIdx, anchorSet, candidatePos)
%COMPUTETDOARESIDUALS Evaluate model minus measured delays for an anchor set.
%   Calculates the error (in meters) between the TDOA *predicted* by the
%   'candidatePos' and the *actual* 'tdoaMeters' that were measured.
    residuals = zeros(numel(anchorSet),1);
    refPos = txPositions(refIdx,:);
    for idx = 1:numel(anchorSet)
        k = anchorSet(idx);
        % Predicted range difference
        modelRange = norm(txPositions(k,:) - candidatePos(:)') - norm(refPos - candidatePos(:)');
        % Absolute error (residual)
        residuals(idx) = abs(modelRange - tdoaMeters(k));
    end
end

function xhat = solveTDOA3D_GN(txPositions, tdoaMeters, refIdx, selectedTx, corrNorm, initialGuess)
%SOLVETDOA3D_GN Solve nonlinear TDOA via Gauss-Newton iterations.
%   This is a custom, iterative, non-linear solver.
    if nargin < 6
        initialGuess = [];
    end
    % ... (Setup for Gauss-Newton iterations) ...
    if isempty(initialGuess)
        p = [mean(txPositions(:,1:2)); 1.5]; % Simple guess
    else
        p = initialGuess(:);
    end
    % ... (Iterative loop to minimize residual error) ...
    xhat = p; % Return final position
end

function xhat = solveTDOA3D_lsqnonlin(txPositions, tdoaMeters, refIdx, selectedTx, corrNorm, initialGuess)
%SOLVETDOA3D_LSQNONLIN Refine TDOA solution using MATLAB's lsqnonlin solver.
%   This requires the Optimization Toolbox. It is generally very accurate.
    if nargin < 6
        initialGuess = [];
    end
    % ... (Setup initial guess p0) ...
    
    % Define the residual-and-Jacobian function for the solver
    function [r,J] = tdoa_res_jac(p)
        % ... (Calculate residual 'r' and Jacobian 'J') ...
    end

    % Configure and run the solver
    opts = optimoptions('lsqnonlin','Algorithm','levenberg-marquardt', ...
        'Display','off','SpecifyObjectiveGradient',true, ...
        'FunctionTolerance',1e-9,'StepTolerance',1e-9,'MaxIterations',200, ...
        'OptimalityTolerance',1e-9);
    xhat = lsqnonlin(@tdoa_res_jac, p0, [], [], opts);
    if ~iscolumn(xhat), xhat = xhat(:); end
end

function xhat3 = solveTDOA2D_GN(txPositions, tdoaMeters, refIdx, selectedTx, zFixed, corrNorm, initialGuess)
%SOLVETDOA2D_GN Solve 2D TDOA with Gauss-Newton while fixing the height.
%   This is a 2D version of the 3D Gauss-Newton solver.
    if nargin < 7
        initialGuess = [];
    end
    % ... (Implementation of 2D Gauss-Newton iterative solver) ...
    p = [0;0]; % Final 2D position
    xhat3 = [p(:); zFixed]; % Return [x; y; zFixed]
end

function hasField = cfgHasField(cfg, fieldName)
%CFGHASFIELD Safe property/field existence check for config objects or structs.
    if isa(cfg,'handle')
        props = properties(cfg);
        hasField = any(strcmp(props, fieldName));
    elseif isstruct(cfg)
        hasField = isfield(cfg, fieldName);
    else
        hasField = false;
    end
end

function value = cfgGet(cfg, fieldName, defaultValue)
%CFGGET Safe accessor for config properties/fields with default fallback.
    if nargin < 3
        defaultValue = [];
    end
    if isa(cfg,'handle')
        if cfgHasField(cfg, fieldName)
            value = cfg.(fieldName);
        else
            value = defaultValue;
        end
    elseif isstruct(cfg)
        if isfield(cfg, fieldName)
            value = cfg.(fieldName);
        else
            value = defaultValue;
        end
    else
        value = defaultValue;
    end
end

function [bestModel, inlierIndices] = ransacTDOA(txPositions, tdoaMeters, refIdx, candidateTx, zFixed, maxNumTrials, maxDistance, sampleSize, modelBounds)
%RANSACTDOA Robustly select TDOA inliers via RANSAC on anchor subsets.
%   This is the function that implements the RANSAC algorithm.
    
    bestInlierCount = -1;
    bestModel = []; % The position [x,y,z] from the best-fitting sample
    inlierIndices = false(size(candidateTx));
    
    refPos = txPositions(refIdx,:);
    numCandidates = numel(candidateTx);

    for i = 1:maxNumTrials
        % --- 1. Randomly select a minimal sample ---
        % (e.g., 3 random anchors from the 'candidateTx' list)
        randIndices = randperm(numCandidates, sampleSize);
        sampleTx = candidateTx(randIndices);
        
        % --- 2. Fit a model to *only* the sample ---
        % (e.g., solve for position using only those 3 anchors)
        txPosForFit = [refPos; txPositions(sampleTx,:)];
        tdoaForFit = [0; tdoaMeters(sampleTx)];
        
        % Use a simple, fast solver for this
        currentModel = properHyperbolicPositioning(txPosForFit, tdoaForFit, 1, (2:sampleSize+1)', [], zFixed, modelBounds);
        
        if isempty(currentModel) || any(~isfinite(currentModel))
            continue; % This random sample was bad (e.g., all in a line)
        end
        
        % --- 3. Find inliers ---
        % Check how many *other* anchors (not in the sample)
        % agree with this 'currentModel'.
        distToAnchors = vecnorm(txPositions(candidateTx,:) - currentModel', 2, 2);
        distToRef = norm(refPos - currentModel');
        % Calculate the predicted TDOA (in meters) for all candidates
        predictedTDOA = distToAnchors - distToRef;
        % Calculate the error (residual)
        residuals = abs(predictedTDOA - tdoaMeters(candidateTx));
        
        % Find all candidates whose error is less than the threshold
        currentInliers = residuals < maxDistance;
        currentInlierCount = nnz(currentInliers);
        
        % --- 4. Check if this is the best model so far ---
        if currentInlierCount > bestInlierCount
            % Yes, this model has the most "votes" (inliers) so far
            bestInlierCount = currentInlierCount;
            bestModel = currentModel;
            inlierIndices = currentInliers;
        end
    end
end

function x = ukfStateFcn(x, dt)
%UKFSTATEFCN Propagate [x y vx vy] using a constant-velocity transition.
%   This is the state transition function handle for the UKF.
%   It predicts: new_pos = old_pos + old_vel * dt
    if nargin < 2
        dt = 1.0; % Default dt
    end
    F = [1 0 dt 0; ...
         0 1 0  dt; ...
         0 0 1  0; ...
         0 0 0  1];
    x = F * x;
end

function y = ukfMeasFcn(x)
%UKFMEASFCN Measurement model extracting the [x y] position from state.
%   This is the measurement function handle for the UKF.
%   It tells the filter that our *measurement* only observes
%   the first two state variables (x and y).
    H = [1 0 0 0; ...
         0 1 0 0];
    y = H * x;
end

function outStr = localBoolToString(flag)
%LOCALBOOLTOSTRING Render logical scalars as '1' or '0'.
    if flag
        outStr = '1';
    else
        outStr = '0';
    end
end