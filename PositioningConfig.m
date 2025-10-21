classdef PositioningConfig < handle
    %POSITIONINGCONFIG Configuration for 5G positioning simulation
    %   Holds transmitter, environment, waveform, channel, and processing
    %   parameters used by the main simulation.
    
    properties

        enableStaticPositions = true         % Use hardcoded positions below
        enableDebug = true                   % Enable debug prints
        debugLevel = 1                       % 0: off, 1: key, 2: verbose

        % Static transmitter positions [x y z] (m)
        staticTxPositions = [
            -17.0, 47.0, 4.4;
              9.0, 47.0, 4.4;
            -17.0,  3.0, 2.6;
              9.0,  3.0, 2.6;
            -17.0, 33.0, 3.8;
            -12.0, 14.0, 3.0;
              9.0, 33.0, 3.8;
              9.0, 17.0, 3.0;
             -4.0, 39.0, 3.5;
              4.0, 39.0, 3.5;
             -4.0, 13.0, 3.1;
              4.0, 13.0, 3.1;
        ];

        % Cell mapping (indices into staticTxPositions)
        cellA = [1, 5, 9, 12];    % CombSize = 2
        cellB = [2, 6,10, 11];    % CombSize = 4
        cellC = [3, 4, 7, 8 ];    % CombSize = 6

        enableStaticReceiverPosition = true  % Use fixed receiver position
        staticRxPosition = [-1.5, 7.0, 1.6]     % [x, y, z] (m)

        % 3D Model Configuration
        customModelFile = 'train_station.stl'
        modelScaleFactor = 12
        useCustomModel = true
        
        % Transmitter Configuration
        numTx = 12
        txFreq = 3.5e9
        txPower = 43                          % dBm
        
        % Enhanced Height Constraints
        heightConstraints = struct(...
            'minHeight', 2.5, ...
            'maxHeight', 4.5, ...
            'preferredHeight', 3.5, ...
            'heightVariation', 1.0 ...
        )
        
        % Placement Strategy Parameters
        clearance = 1.0
        minSeparation = 2.0
        nearFieldRadius = 25.0
        
        % 5G NR Carrier Configuration
        carrierConfig = struct(...
            'SubcarrierSpacing', 30, ...
            'CyclicPrefix', 'normal', ...
            'NSizeGrid', 273, ...
            'NStartGrid', 0 ...
        )
        
        % Enhanced Multi-Cell PRS Configuration
        prsConfig = struct(...
            'PRSResourceSetPeriod', [40, 0], ...
            'PRSResourceOffset', 0, ...
            'NumRB', 272, ...
            'RBOffset', 0, ...
            'SymbolStart', 2, ...
            'NumPRSSymbols', 12, ...
            'CombSize', 2, ...
            'REOffset', 0, ...
            'PRSResourceRepetition', 16, ...
            'PRSResourceTimeGap', 1, ...
            'NPRSID', 0, ...
            'MutingPattern1', [], ...
            'MutingBitRepetition', 2, ...
            'MutingPattern2', [], ...
            'enableMultiCellCoordination', true, ...
            'combSizeOptions', [2, 4, 6], ...
            'maxPRSID', 4095 ...
        )
        
        % Enhanced features control
        enableDynamicBandwidth = true
        enableAdvancedMuting = false
        enablePowerControl = true
        enableMultiLayer = true
        enableLoadAwareScheduling = true
        enableTimingErrors = true
        enableRaytraceCache = true
        enableUKFTracking = false
        parallelProcessing = struct(...
            'enable', true, ...
            'enableWaveformParallel', true, ...
            'enableChannelParallel', true, ...
            'maxChannelWorkers', 1, ...
            'channelMemoryGuardGB', 24, ...
            'maxWorkers', 12, ...
            'targetMemoryPerWorkerGB', 8, ...
            'reserveMemoryGB', 32, ...
            'autoShutdown', false)
        useSinglePrecisionBuffers = true
        
        % Advanced parameters
        targetSINR = 15
        timingErrorVariance = 10e-9
        loadThresholds = [30, 70]          % Low/high load thresholds
        maxPowerAdjustment = 6             % dB maximum power change
        
        % Multi-layer positioning configuration
        frequencyLayers = struct(...
            'numLayers', 3, ...
            'baseFrequency', 3.5e9, ...
            'frequencyOffsets', [0, -0.5e9, 0.5e9], ...
            'bandwidthRatios', [1.0, 0.5, 0.75], ...
            'priorities', [1, 2, 3] ...
        )
        
        % Network load simulation parameters
        networkLoad = struct(...
            'enableSimulation', true, ...
            'baseUtilization', 50, ...      % Base network utilization %
            'utilizationVariance', 20, ...   % Variation in utilization
            'loadUpdatePeriod', 10 ...       % Slots between load updates
        )
        
        % Channel condition assessment
        channelAssessment = struct(...
            'enableCSIFeedback', true, ...
            'measurementPeriod', 5, ...      % Slots between measurements
            'sinrThresholds', [5, 15, 25], ...  % Poor/Good/Excellent thresholds
            'interferenceThreshold', 0.5 ...  % Normalized interference level
        )
        
        % Ray Tracing Configuration
        rayTracingConfig = struct(...
            'MaxNumReflections', 2, ...        % Maximum reflections
            'MaxNumDiffractions', 1, ...       % Maximum diffractions
            'SurfaceMaterial', 'concrete', ... % Surface material
            'CoordinateSystem', 'cartesian' ...
        )
        
        % Channel Modeling Parameters
        channelConfig = struct(...
            'pathLossExponent', 2.0, ...       % Improved for enhanced placement
            'shadowingStd', 6, ...             % Reduced shadowing variance
            'excessDelayMax', 30e-9, ...       % Maximum excess delay (30ns)
            'SNR_dB', 25 ...                   % Enhanced SNR for better placement
        )


        % Waveform Generation
        numSlots = 56
        
        % TDOA Processing
        tdoaConfig = struct( ...
            'maxSamples', 200000, ...
            'maxLag', 80000, ...
            'correlationThreshold', 0.001, ...
            'peakQualityThreshold', 0.42, ...
            'useGCCPHAT', false, ...              % (optional future)
            'peakStrategy', 'leading-edge', ...
            'delayWindowMarginNs', 2000, ...      % = 2.0 us half-window
            'expectedGateMeters', 30, ...         % base gate; function ensures ≥ 4 samples
            'leadingEdgeAlpha', 3.5, ...          % CFAR multiplier
            'leadingEdgeHoldSamples', 3, ...      % stability
            'topKRef', 4, ...                     % ref selection pool
            'bestK', 6, ...
            'maxGDOP', 6, ...
            'minAnchorAngleDeg', 28 ...
        );

        
        % Solver control
        force2DPositioning = true          % Force 2D (fix z) positioning by default
        defaultUEHeight = 1.8              % Used when z is fixed and no staticRxPosition

        % Performance Thresholds
        performanceThresholds = struct(...
            'excellent', 1.0, ...             % < 1m error
            'veryGood', 2.0, ...              % < 2m error
            'good', 3.0, ...                  % < 3m error
            'acceptable', 5.0 ...             % < 5m error
        )

        % Position sweep experimentation
        positionSweep = struct( ...
            'enable', true, ...
            'numPoints', 10, ...
            'samplesPerPoint', 5, ...
            'startPoint', [-1.5, 7.0, 1.6], ...
            'endPoint',   [-1.5, 35.0, 1.6] ...
        )

        % SNR sweep experimentation
        snrSweep = struct( ...
            'enable', true, ...
            'values', [25], ...
            'ukfModes', [false, true], ...
            'tagPrefix', 'snrRun' ...
        )

        % Result logging configuration
        resultLogging = struct( ...
            'enable', true, ...
            'outputFile', 'positioning_sweep_results.csv', ...
            'overwrite', true ...
        )
    end
    
    methods
        function obj = PositioningConfig()
            %POSITIONINGCONFIG Constructor
            fprintf('Initializing Enhanced 5G Positioning Configuration...\n');
            obj.validateConfiguration();
        end
        
        function validateConfiguration(obj)
            %VALIDATECONFIGURATION Validate configuration parameters
            
            % Validate transmitter parameters
            assert(obj.numTx >= 4, 'Minimum 4 transmitters required for 3D positioning');
            assert(obj.txFreq > 0, 'Transmitter frequency must be positive');
            assert(obj.txPower > 0, 'Transmitter power must be positive');
            
            % Validate height constraints
            assert(obj.heightConstraints.minHeight < obj.heightConstraints.maxHeight, ...
                'Minimum height must be less than maximum height');
            assert(obj.heightConstraints.preferredHeight >= obj.heightConstraints.minHeight && ...
                   obj.heightConstraints.preferredHeight <= obj.heightConstraints.maxHeight, ...
                'Preferred height must be within min/max range');
            
            % Validate placement parameters
            assert(obj.clearance > 0, 'Clearance must be positive');
            assert(obj.minSeparation > 0, 'Minimum separation must be positive');
            assert(obj.nearFieldRadius > 0, 'Near field radius must be positive');
            
            % Enhanced features validation
            assert(obj.targetSINR > 0 && obj.targetSINR < 50, ...
                'Target SINR must be between 0 and 50 dB');
            assert(obj.timingErrorVariance > 0, ...
                'Timing error variance must be positive');
            assert(length(obj.loadThresholds) == 2 && obj.loadThresholds(1) < obj.loadThresholds(2), ...
                'Load thresholds must be [low, high] with low < high');
            assert(obj.frequencyLayers.numLayers <= 5, ...
                'Maximum 5 frequency layers supported');
            
            if isstruct(obj.parallelProcessing)
                if ~isfield(obj.parallelProcessing, 'maxWorkers') || obj.parallelProcessing.maxWorkers < 1
                    obj.parallelProcessing.maxWorkers = 1;
                end
                if ~isfield(obj.parallelProcessing, 'maxChannelWorkers') || obj.parallelProcessing.maxChannelWorkers < 1
                    obj.parallelProcessing.maxChannelWorkers = 1;
                end
                if obj.parallelProcessing.maxChannelWorkers > obj.parallelProcessing.maxWorkers
                    obj.parallelProcessing.maxChannelWorkers = obj.parallelProcessing.maxWorkers;
                end
                if ~isfield(obj.parallelProcessing, 'targetMemoryPerWorkerGB') || obj.parallelProcessing.targetMemoryPerWorkerGB <= 0
                    obj.parallelProcessing.targetMemoryPerWorkerGB = 24;
                end
                if ~isfield(obj.parallelProcessing, 'reserveMemoryGB') || obj.parallelProcessing.reserveMemoryGB < 0
                    obj.parallelProcessing.reserveMemoryGB = 16;
                end
                if ~isfield(obj.parallelProcessing, 'channelMemoryGuardGB') || obj.parallelProcessing.channelMemoryGuardGB <= 0
                    obj.parallelProcessing.channelMemoryGuardGB = 20;
                end
            end

            % Position sweep validation
            if obj.positionSweep.enable
                assert(obj.positionSweep.numPoints >= 2, 'Position sweep requires at least two points.');
                assert(numel(obj.positionSweep.startPoint) == 3, 'Start point must be a 3-element vector.');
                assert(numel(obj.positionSweep.endPoint) == 3, 'End point must be a 3-element vector.');
                assert(obj.positionSweep.samplesPerPoint >= 1, 'Samples per point must be >= 1.');
            end

            if obj.snrSweep.enable
                assert(~isempty(obj.snrSweep.values), 'SNR sweep requires at least one SNR value.');
                assert(isnumeric(obj.snrSweep.values), 'SNR sweep values must be numeric.');
                if isempty(obj.snrSweep.ukfModes)
                    obj.snrSweep.ukfModes = logical(obj.enableUKFTracking);
                elseif isnumeric(obj.snrSweep.ukfModes)
                    assert(all(ismember(obj.snrSweep.ukfModes, [0 1])), 'SNR sweep ukfModes numeric values must be 0 or 1.');
                    obj.snrSweep.ukfModes = logical(obj.snrSweep.ukfModes);
                else
                    assert(islogical(obj.snrSweep.ukfModes), 'SNR sweep ukfModes must be logical or numeric 0/1.');
                end
            end

            if obj.resultLogging.enable
                assert(ischar(obj.resultLogging.outputFile) || isstring(obj.resultLogging.outputFile), ...
                    'Result logging outputFile must be a character vector or string.');
                if ~isfield(obj.resultLogging, 'overwrite')
                    obj.resultLogging.overwrite = true;
                end
            end
            
            fprintf('Enhanced configuration validation successful.\n');
        end
        
        function displayConfiguration(obj)
            %DISPLAYCONFIGURATION Display current configuration
            
            fprintf('\n=== Enhanced 5G Positioning Configuration ===\n');
            fprintf('Model: %s (scale: %.1fx)\n', obj.customModelFile, obj.modelScaleFactor);
            fprintf('Transmitters: %d @ %.1fGHz, %.0fdBm\n', obj.numTx, obj.txFreq/1e9, obj.txPower);
            fprintf('Height range: %.1f - %.1fm (preferred: %.1fm)\n', ...
                obj.heightConstraints.minHeight, obj.heightConstraints.maxHeight, ...
                obj.heightConstraints.preferredHeight);
            fprintf('Near field radius: %.1fm\n', obj.nearFieldRadius);
            fprintf('Minimum separation: %.1fm\n', obj.minSeparation);
            fprintf('SNR: %.0fdB\n', obj.channelConfig.SNR_dB);
            fprintf('Correlation threshold: %.2f\n', obj.tdoaConfig.correlationThreshold);
            
            % Enhanced features status
            fprintf('\nEnhanced Features:\n');
            fprintf('  Dynamic Bandwidth: %s\n', obj.boolToString(obj.enableDynamicBandwidth));
            fprintf('  Advanced Muting: %s\n', obj.boolToString(obj.enableAdvancedMuting));
            fprintf('  Power Control: %s (Target SINR: %.0f dB)\n', obj.boolToString(obj.enablePowerControl), obj.targetSINR);
            fprintf('  Multi-Layer: %s (%d layers)\n', obj.boolToString(obj.enableMultiLayer), obj.frequencyLayers.numLayers);
            fprintf('  Load-Aware Scheduling: %s\n', obj.boolToString(obj.enableLoadAwareScheduling));
            fprintf('  Timing Errors: %s (%.0f ns RMS)\n', obj.boolToString(obj.enableTimingErrors), obj.timingErrorVariance*1e9);
            fprintf('  Ray-Trace Cache: %s\n', obj.boolToString(obj.enableRaytraceCache));
            
            if isstruct(obj.parallelProcessing) && obj.parallelProcessing.enable
                fprintf('\nParallel Processing:\n');
                fprintf('  Waveform parallel: %s\n', obj.boolToString(obj.parallelProcessing.enableWaveformParallel));
                fprintf('  Channel parallel: %s\n', obj.boolToString(obj.parallelProcessing.enableChannelParallel));
                fprintf('  Max workers: %d (target %.0f GB per worker, reserve %.0f GB)\n', ...
                    obj.parallelProcessing.maxWorkers, ...
                    obj.parallelProcessing.targetMemoryPerWorkerGB, ...
                    obj.parallelProcessing.reserveMemoryGB);
            fprintf('  Channel concurrency cap: %d (guard %.0f GB per worker)\n', ...
                    obj.parallelProcessing.maxChannelWorkers, ...
                    obj.parallelProcessing.channelMemoryGuardGB);
            else
                fprintf('\nParallel Processing: DISABLED\n');
            end
            
            fprintf('\nBuffer Precision:\n');
            fprintf('  Single-precision buffers: %s\n', obj.boolToString(obj.useSinglePrecisionBuffers));
            fprintf('\nPosition Solver:\n');
            fprintf('  Force 2D positioning: %s\n', obj.boolToString(obj.force2DPositioning));
            fprintf('  Default UE height: %.2f m\n', obj.defaultUEHeight);
            fprintf('  Min anchor separation: %.1f deg\n', obj.tdoaConfig.minAnchorAngleDeg);
            fprintf('  UKF tracking: %s\n', obj.boolToString(obj.enableUKFTracking));

            if obj.positionSweep.enable
                fprintf('\nPosition Sweep:\n');
                fprintf('  Points: %d, Samples/Point: %d\n', obj.positionSweep.numPoints, obj.positionSweep.samplesPerPoint);
                fprintf('  Start: [%.2f %.2f %.2f] m\n', obj.positionSweep.startPoint);
                fprintf('  End:   [%.2f %.2f %.2f] m\n', obj.positionSweep.endPoint);
            end

            if obj.snrSweep.enable
                fprintf('\nSNR Sweep:\n');
                fprintf('  Values (dB): %s\n', mat2str(obj.snrSweep.values));
                fprintf('  UKF modes: %s\n', mat2str(logical(obj.snrSweep.ukfModes)));
                if isfield(obj.snrSweep,'tagPrefix') && ~isempty(obj.snrSweep.tagPrefix)
                    fprintf('  Tag prefix: %s\n', obj.snrSweep.tagPrefix);
                end
            end

            if obj.resultLogging.enable
                fprintf('\nResult Logging:\n');
                fprintf('  Output file: %s (%s)\n', obj.resultLogging.outputFile, ...
                    obj.ternary(obj.resultLogging.overwrite, 'overwrite', 'append'));
            end
        end
        
        function str = boolToString(~, boolVal)
            %BOOLTOSTRING Convert boolean to string for display
            if boolVal
                str = 'ENABLED';
            else
                str = 'DISABLED';
            end
        end
        
        function out = ternary(~, condition, trueVal, falseVal)
            if condition
                out = trueVal;
            else
                out = falseVal;
            end
        end
    end
end
