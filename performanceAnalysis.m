function performanceAnalysis(txPositions, rxPosition, propagationLoss, maxCorr, estimatedPosition, positionError, roomDims, config)
%PERFORMANCEANALYSIS Analyze coverage, geometry, and expected accuracy
% Inputs:
%   txPositions       - [N x 3] transmitter positions (m)
%   rxPosition        - [1 x 3] receiver true position (m)
%   propagationLoss   - [N x 1] per-Tx path loss (dB)
%   maxCorr           - [N x 1] per-Tx corr_norm quality [0,1]
%   estimatedPosition - [1 x 3] estimated position (m)
%   positionError     - scalar horizontal error (m)
%   roomDims          - [L W H] environment dimensions (m)
%   config            - configuration struct/class
% Output: (prints summary to console)
    
    fprintf('\n=== ENHANCED PERFORMANCE ANALYSIS ===\n');
    
    numTx = size(txPositions, 1);
    
    % Distance analysis
    analyzeDistanceDistribution(txPositions, rxPosition, config.nearFieldRadius);

    
    % Signal strength analysis
    analyzeSignalStrength(propagationLoss, numTx);
    
    % Geometry analysis with error handling
    try
        analyzeGeometry(txPositions, rxPosition);
    catch ME
        fprintf('Geometry analysis failed: %s\n', ME.message);
        fprintf('Skipping geometry analysis...\n');
    end
    
    % Performance prediction with error handling
    try
        predictPerformance(txPositions, rxPosition, roomDims);
    catch ME
        fprintf('Performance prediction failed: %s\n', ME.message);
        fprintf('Skipping performance prediction...\n');
    end
    
    % Coverage analysis
    analyzeCoverage(txPositions, rxPosition, config.nearFieldRadius, numTx);
    
    % Improvement suggestions intentionally omitted from console output
end

function analyzeDistanceDistribution(txPositions, rxPosition, nearFieldRadius)
    %ANALYZEDISTANCEDISTRIBUTION Analyze transmitter distance distribution

    numTx = size(txPositions, 1);

    % Calculate 3D distances from receiver to each transmitter
    distances = vecnorm(txPositions - rxPosition, 2, 2);

    nearFieldTx = sum(distances <= nearFieldRadius);
    farFieldTx  = sum(distances >  nearFieldRadius);

    fprintf('Transmitter distance distribution (3D):\n');
    fprintf('  Near-field (<=%.0fm): %d transmitters\n', nearFieldRadius, nearFieldTx);
    fprintf('  Far-field (>%.0fm): %d transmitters\n', nearFieldRadius, farFieldTx);
    fprintf('  Closest transmitter: %.1fm\n', min(distances));
    fprintf('  Furthest transmitter: %.1fm\n', max(distances));
    fprintf('  Average distance: %.1fm\n', mean(distances));
    fprintf('  Distance std deviation: %.1fm\n', std(distances));
end
function analyzeSignalStrength(propagationLoss, numTx)
    %ANALYZESIGNALSTRENGTH Analyze signal strength characteristics
    
    finiteMask    = isfinite(propagationLoss);
    plFinite      = propagationLoss(finiteMask);
    if isempty(plFinite)
        avgPathLoss = NaN; minPathLoss = NaN; maxPathLoss = NaN;
        strongSignals = 0; weakSignals = 0; count = 0;
    else
        avgPathLoss   = mean(plFinite);
        minPathLoss   = min(plFinite);
        maxPathLoss   = max(plFinite);
        strongSignals = sum(plFinite <= 85);
        weakSignals   = sum(plFinite > 95);
        count         = numel(plFinite);
    end
    
    fprintf('Signal strength analysis:\n');
    fprintf('  Average path loss: %.1f dB\n', avgPathLoss);
    fprintf('  Path loss range: %.1f - %.1f dB\n', minPathLoss, maxPathLoss);
    if count>0
        fprintf('  Strong signals (<=85 dB): %d/%d (%.1f%%)\n', strongSignals, numTx, strongSignals/numTx*100);
        fprintf('  Weak signals (>95 dB): %d/%d (%.1f%%)\n', weakSignals, numTx, weakSignals/numTx*100);
    else
        fprintf('  Strong signals (<=85 dB): %d/%d (%.1f%%)\n', 0, numTx, 0);
        fprintf('  Weak signals (>95 dB): %d/%d (%.1f%%)\n', 0, numTx, 0);
    end
    
    % Signal quality assessment
    if ~isfinite(avgPathLoss)
        fprintf('  Signal quality: Unknown (insufficient data)\n');
    elseif avgPathLoss < 80
        fprintf('  Signal quality: Excellent\n');
    elseif avgPathLoss < 90
        fprintf('  Signal quality: Good\n');
    elseif avgPathLoss < 100
        fprintf('  Signal quality: Fair\n');
    else
        fprintf('  Signal quality: Poor\n');
    end
end

function analyzeGeometry(txPositions, rxPosition)
    %ANALYZEGEOMETRY Analyze positioning geometry
    
    % GDOP and geometry analysis
    finalGDOP = calculateGDOP(txPositions, rxPosition);
    avgAngularSep = calculateAverageAngularSeparation(txPositions, rxPosition);
    
    fprintf('Geometry analysis:\n');
    fprintf('  GDOP: %.2f\n', finalGDOP);
    fprintf('  Average angular separation: %.1f degrees\n', avgAngularSep);
    
    % Geometry quality assessment
    if finalGDOP < 5
        fprintf('  Geometry quality: Excellent\n'); gdopRating = 1;
    elseif finalGDOP < 10
        fprintf('  Geometry quality: Good\n'); gdopRating = 2;
    elseif finalGDOP < 20
        fprintf('  Geometry quality: Fair\n'); gdopRating = 3;
    else
        fprintf('  Geometry quality: Poor\n'); gdopRating = 4;
    end
    
    % Angular diversity assessment
    if avgAngularSep > 45
        fprintf('  Angular diversity: Excellent\n'); angRating = 1;
    elseif avgAngularSep > 30
        fprintf('  Angular diversity: Good\n'); angRating = 2;
    elseif avgAngularSep > 20
        fprintf('  Angular diversity: Fair\n'); angRating = 3;
    else
        fprintf('  Angular diversity: Poor\n'); angRating = 4;
    end

    % Harmonized summary rating: choose the worse (more conservative)
    ratings = { 'Excellent', 'Good', 'Fair', 'Poor' };
    combined = ratings{max(gdopRating, angRating)};
    fprintf('  Combined geometry rating: %s\n', combined);
end

function predictPerformance(txPositions, rxPosition, roomDims)
    %PREDICTPERFORMANCE Predict positioning performance
    
    finalGDOP = calculateGDOP(txPositions, rxPosition);
    
    % Calculate minimum 3D distance
    distances = vecnorm(txPositions - rxPosition, 2, 2);
    minDistance = min(distances);
    
    avgAngularSep = calculateAverageAngularSeparation(txPositions, rxPosition);
    
    % Performance prediction
    expectedAccuracy = predictPositioningAccuracy(finalGDOP, minDistance, avgAngularSep);
    
    fprintf('Performance prediction:\n');
    if expectedAccuracy < 3
        fprintf('  Expected accuracy: %.1fm (EXCELLENT for large space)\n', expectedAccuracy);
    elseif expectedAccuracy < 5
        fprintf('  Expected accuracy: %.1fm (VERY GOOD for large space)\n', expectedAccuracy);
    elseif expectedAccuracy < 10
        fprintf('  Expected accuracy: %.1fm (GOOD for large space)\n', expectedAccuracy);
    else
        fprintf('  Expected accuracy: %.1fm (NEEDS OPTIMIZATION)\n', expectedAccuracy);
    end
end

function analyzeCoverage(txPositions, rxPosition, nearFieldRadius, numTx)
    %ANALYZECOVERAGE Analyze coverage efficiency
    
    distances = vecnorm(txPositions - rxPosition, 2, 2);
    
    nearFieldTx = sum(distances <= nearFieldRadius);
    coverageEfficiency = nearFieldTx / numTx * 100;
    
    fprintf('Coverage analysis:\n');
    fprintf('  Coverage efficiency: %.1f%% (transmitters in near-field)\n', coverageEfficiency);
    
    if coverageEfficiency > 70
        fprintf('  Coverage quality: Excellent\n');
    elseif coverageEfficiency > 50
        fprintf('  Coverage quality: Good\n');
    elseif coverageEfficiency > 30
        fprintf('  Coverage quality: Fair\n');
    else
        fprintf('  Coverage quality: Poor\n');
    end
end

function accuracy = predictPositioningAccuracy(gdop, minDistance, avgAngularSep)
    %PREDICTPOSITIONINGACCURACY Predict positioning accuracy for large spaces
    %   Empirical model for large indoor spaces
    
    % Base accuracy from GDOP
    gdopFactor = gdop * 0.25; % Improved ranging accuracy assumption
    
    % Distance factor (penalty for far transmitters)
    distFactor = 1 + max(0, (minDistance - 15) / 30); % Penalty starts at 15m
    
    % Angular diversity factor
    angularFactor = 1 + max(0, (45 - avgAngularSep) / 45); % Penalty for <45 deg separation
    % Large space penalty
    largeSpaceFactor = 1.2; % 20% penalty for large indoor environments
    
    accuracy = gdopFactor * distFactor * angularFactor * largeSpaceFactor;
end

%% CODE HYGIENE FIX: Replaced with the robust, correct GDOP function
function gdop = calculateGDOP(txPositions, rxPosition)
%CALCULATEGDOP FIXED - TDOA-based GDOP using proper Jacobian matrix
%   This is the correct implementation based on the proper TDOA Jacobian.

% Input validation
numTx = size(txPositions, 1);
if numTx < 4
    gdop = inf; 
    return;
end

if isempty(rxPosition) || any(~isfinite(rxPosition))
    gdop = inf;
    return;
end

% Ensure rxPosition is a row vector for broadcasting
if size(rxPosition, 1) > 1
    rxPosition = rxPosition(:)';
end

try
    % Calculate unit vectors from RX to each TX (proper TDOA geometry)
    u = txPositions - rxPosition;
    
    % Calculate distances and handle potential zeros
    d = vecnorm(u, 2, 2); % Column vector of distances
    
    % Check for degenerate cases
    if any(d < 1e-6)
        gdop = inf;
        return;
    end
    
    % Normalize to unit vectors
    u = u ./ d;
    
    % Build proper TDOA Jacobian matrix (relative to TX1 as reference)
    H = u(2:end, :) - u(1, :); % (N-1)×3 matrix
    
    if rank(H) < 3
        gdop = inf;
        return;
    end
    
    % Calculate GDOP using proper TDOA covariance matrix
    try
        HTH = H.' * H;
        if cond(HTH) > 1e12
            gdop = inf;
            return;
        end
        Q = inv(HTH);
        gdop = sqrt(trace(Q));
        if ~isfinite(gdop) || gdop <= 0
            gdop = inf;
        end
    catch
        gdop = inf;
    end
catch
    gdop = inf;
end
end

function avgAngle = calculateAverageAngularSeparation(txPositions, rxPosition)
    %CALCULATEAVERAGEANGULARSEPARATION Calculate average angular separation
    %   This function calculates the average angular separation between
    %   transmitters as seen from the receiver position.
    
    angles = [];
    for i = 1:size(txPositions, 1)
        angle = atan2d(txPositions(i, 2) - rxPosition(2), txPositions(i, 1) - rxPosition(1));
        angles = [angles; angle];
    end
    
    angles = sort(angles);
    angleDiffs = diff([angles; angles(1) + 360]);
    avgAngle = mean(angleDiffs);
end


