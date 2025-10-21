function txPositions = enhancedTransmitterPlacement(roomBounds, numTx, rxPosition, heightConstraints)
    %ENHANCEDTRANSMITTERPLACEMENT Transmitter placement for indoor spaces
    % Inputs:
    %   roomBounds        - struct with xmin/xmax/ymin/ymax/zmin/zmax (m)
    %   numTx             - number of transmitters
    %   rxPosition        - receiver position [x y z] (m)
    %   heightConstraints - struct with minHeight/maxHeight etc.
    % Output:
    %   txPositions       - [numTx x 3] positions (m)

    % Room dimensions
    roomWidth = roomBounds.xmax - roomBounds.xmin;
    roomLength = roomBounds.ymax - roomBounds.ymin;
    roomHeight = roomBounds.zmax - roomBounds.zmin;
    
    % Enhanced parameters for large spaces
    clearance = 1.0;  % Reduced clearance for better coverage
    minSeparation = 2.0;  % Reduced from 3.0m for denser placement
    nearFieldRadius = 25.0;  % Ensure transmitters within 25m of any point
    
    % Step 1: Analyze receiver position
    receiverAnalysis = analyzeReceiverLocation(rxPosition, roomBounds);
    
    % Step 2: Create distance-based priority zones
    zones = createDistanceBasedZones(roomBounds, rxPosition, nearFieldRadius, clearance);
    
    % Step 3: Apply placement strategy based on receiver location
    if receiverAnalysis.isCorner
        txPositions = cornerAwarePlacement(zones, receiverAnalysis, numTx, heightConstraints, minSeparation);
    else
        txPositions = centerRegionPlacement(zones, numTx, heightConstraints, minSeparation, rxPosition);
    end
    
    % Step 4: Geometry optimization
    txPositions = optimizeGeometry(txPositions, rxPosition, roomBounds, heightConstraints, minSeparation);
    
    % Step 5: Validation and metrics
    validatePlacement(txPositions, rxPosition, roomBounds, nearFieldRadius);
    
    % Placement complete
end

%% ========================================================================
%% SUPPORT FUNCTIONS - All integrated to avoid dependency issues
%% ========================================================================

function analysis = analyzeReceiverLocation(rxPosition, roomBounds)
    %ANALYZERECEIVERPOSITION Analyze receiver location relative to room
    
    roomWidth = roomBounds.xmax - roomBounds.xmin;
    roomLength = roomBounds.ymax - roomBounds.ymin;
    
    % Calculate distances to walls
    distToLeft = rxPosition(1) - roomBounds.xmin;
    distToRight = roomBounds.xmax - rxPosition(1);
    distToBottom = rxPosition(2) - roomBounds.ymin;
    distToTop = roomBounds.ymax - rxPosition(2);
    
    % Determine position type
    cornerThreshold = min(roomWidth, roomLength) * 0.2; % 20% of room dimension
    edgeThreshold = min(roomWidth, roomLength) * 0.3;   % 30% of room dimension
    
    analysis.distToWalls = [distToLeft, distToRight, distToBottom, distToTop];
    analysis.minDistToWall = min(analysis.distToWalls);
    analysis.maxDistToWall = max(analysis.distToWalls);
    
    if analysis.minDistToWall < cornerThreshold
        analysis.isCorner = true;
        analysis.isEdge = true;
        analysis.positionType = 'corner';
        
        % Determine which corner
        if distToLeft < cornerThreshold && distToBottom < cornerThreshold
            analysis.cornerType = 'bottom-left';
        elseif distToRight < cornerThreshold && distToBottom < cornerThreshold
            analysis.cornerType = 'bottom-right';
        elseif distToLeft < cornerThreshold && distToTop < cornerThreshold
            analysis.cornerType = 'top-left';
        else
            analysis.cornerType = 'top-right';
        end
        
    elseif analysis.minDistToWall < edgeThreshold
        analysis.isCorner = false;
        analysis.isEdge = true;
        analysis.positionType = 'edge';
        analysis.cornerType = 'none';
    else
        analysis.isCorner = false;
        analysis.isEdge = false;
        analysis.positionType = 'center';
        analysis.cornerType = 'none';
    end
    
    analysis.description = sprintf('%s region (%.1fm from nearest wall)', ...
        analysis.positionType, analysis.minDistToWall);
end

function zones = createDistanceBasedZones(roomBounds, rxPosition, nearFieldRadius, clearance)
    %CREATEDISTANCEBASEDZONES Create priority zones for transmitter placement
    
    
    % Zone 1: Immediate vicinity (Highest Priority)
    immediateRadius = min(15, nearFieldRadius * 0.6);
    zones.immediate = createCircularZone(rxPosition, immediateRadius, roomBounds, clearance);
    zones.immediate.priority = 1.0;
    zones.immediate.maxTx = 3;
    zones.immediate.name = 'immediate';
    
    % Zone 2: Near field (High Priority)
    zones.nearField = createRingZone(rxPosition, immediateRadius, nearFieldRadius, roomBounds, clearance);
    zones.nearField.priority = 0.8;
    zones.nearField.maxTx = 4;
    zones.nearField.name = 'nearField';
    
    % Zone 3: Far field (Medium Priority)
    zones.farField = createFarFieldZone(rxPosition, nearFieldRadius, roomBounds, clearance);
    zones.farField.priority = 0.4;
    zones.farField.maxTx = 3;
    zones.farField.name = 'farField';
    
    % Zone 4: Opposite diagonal (Special Priority for GDOP)
    zones.oppositeDiagonal = createOppositeZone(rxPosition, roomBounds, clearance);
    zones.oppositeDiagonal.priority = 0.9;
    zones.oppositeDiagonal.maxTx = 2;
    zones.oppositeDiagonal.name = 'oppositeDiagonal';
    
    % Debug zone information
    zoneNames = fieldnames(zones);
    for i = 1:length(zoneNames)
        zone = zones.(zoneNames{i});
        % zone summary suppressed
    end
end

function zone = createCircularZone(center, radius, roomBounds, clearance)
    %CREATECIRCULARZONE Create circular zone for transmitter placement
    
    zone.type = 'circular';
    zone.center = center(1:2);
    zone.radius = radius;
    zone.bounds = [
        max(roomBounds.xmin + clearance, center(1) - radius), ...
        min(roomBounds.xmax - clearance, center(1) + radius), ...
        max(roomBounds.ymin + clearance, center(2) - radius), ...
        min(roomBounds.ymax - clearance, center(2) + radius)
    ];
end

function zone = createRingZone(center, innerRadius, outerRadius, roomBounds, clearance)
    %CREATERINGZONE Create ring zone for transmitter placement
    
    zone.type = 'ring';
    zone.center = center(1:2);
    zone.innerRadius = innerRadius;
    zone.outerRadius = outerRadius;
    zone.bounds = [
        max(roomBounds.xmin + clearance, center(1) - outerRadius), ...
        min(roomBounds.xmax - clearance, center(1) + outerRadius), ...
        max(roomBounds.ymin + clearance, center(2) - outerRadius), ...
        min(roomBounds.ymax - clearance, center(2) + outerRadius)
    ];
end

function zone = createFarFieldZone(center, nearFieldRadius, roomBounds, clearance)
    %CREATEFARFIELDZONE Create far field zone for transmitter placement
    
    zone.type = 'farField';
    zone.center = center(1:2);
    zone.exclusionRadius = nearFieldRadius;
    zone.bounds = [
        roomBounds.xmin + clearance, roomBounds.xmax - clearance, ...
        roomBounds.ymin + clearance, roomBounds.ymax - clearance
    ];
end

function zone = createOppositeZone(rxPosition, roomBounds, clearance)
    %CREATEOPPOSITEZONE Create opposite zone for GDOP optimization
    
    roomCenterX = (roomBounds.xmin + roomBounds.xmax) / 2;
    roomCenterY = (roomBounds.ymin + roomBounds.ymax) / 2;
    
    % Find opposite point
    oppositeX = 2 * roomCenterX - rxPosition(1);
    oppositeY = 2 * roomCenterY - rxPosition(2);
    
    % Clamp to room bounds
    oppositeX = max(roomBounds.xmin + clearance, min(roomBounds.xmax - clearance, oppositeX));
    oppositeY = max(roomBounds.ymin + clearance, min(roomBounds.ymax - clearance, oppositeY));
    
    % Create zone around opposite point
    zoneRadius = min(roomBounds.xmax - roomBounds.xmin, roomBounds.ymax - roomBounds.ymin) * 0.25;
    
    zone.type = 'circular';
    zone.center = [oppositeX, oppositeY];
    zone.radius = zoneRadius;
    zone.bounds = [
        max(roomBounds.xmin + clearance, oppositeX - zoneRadius), ...
        min(roomBounds.xmax - clearance, oppositeX + zoneRadius), ...
        max(roomBounds.ymin + clearance, oppositeY - zoneRadius), ...
        min(roomBounds.ymax - clearance, oppositeY + zoneRadius)
    ];
end

function txPositions = cornerAwarePlacement(zones, receiverAnalysis, numTx, heightConstraints, minSeparation)
    %CORNERAWAREPLACEMENT Corner-aware placement strategy
    
    % Corner-aware placement
    
    txPositions = [];
    
    % Strategy 1: L-shaped placement around corner
    lShapePositions = createLShapePlacement(receiverAnalysis, zones, heightConstraints, numTx);
    txPositions = addValidPositions(txPositions, lShapePositions, minSeparation);
    
    % Strategy 2: Fill remaining with distance-prioritized placement
    remainingTx = numTx - size(txPositions, 1);
    if remainingTx > 0
        additionalPositions = fillWithDistancePriority(zones, txPositions, remainingTx, heightConstraints, minSeparation);
        txPositions = [txPositions; additionalPositions];
    end
    
    % summary suppressed
end

function txPositions = centerRegionPlacement(zones, numTx, heightConstraints, minSeparation, rxPosition)
    %CENTERREGIONPLACEMENT Center region placement strategy
    
    % Center-region placement
    
    txPositions = [];
    
    % For center receivers, use radial placement strategy
    zoneOrder = {'immediate', 'nearField', 'oppositeDiagonal', 'farField'};
    
    for i = 1:length(zoneOrder)
        zoneName = zoneOrder{i};
        if ~isfield(zones, zoneName)
            continue;
        end
        
        zone = zones.(zoneName);
        remainingTx = numTx - size(txPositions, 1);
        
        if remainingTx <= 0
            break;
        end
        
        % Determine how many transmitters for this zone
        txForZone = min(zone.maxTx, remainingTx);
        
        if txForZone > 0
            % placing summary suppressed
            zonePositions = generatePositionsInZone(zone, txForZone, txPositions, heightConstraints, minSeparation);
            txPositions = [txPositions; zonePositions];
        end
    end
    
    % summary suppressed
end

function positions = createLShapePlacement(receiverAnalysis, zones, heightConstraints, numTx)
    %CREATELSHAPEPLACEMENT Create L-shaped placement around corner
    
    positions = [];
    
    % Target 60% of transmitters in L-shape around corner
    lShapeTx = max(3, round(numTx * 0.6));
    
    % Get immediate zone for L-shape
    immediateZone = zones.immediate;
    
    % Create L-shaped grid based on corner type
    switch receiverAnalysis.cornerType
        case 'bottom-left'
            rightArm = [
                immediateZone.bounds(1) + 3, immediateZone.bounds(3) + 1, applyHeightConstraints(heightConstraints);
                immediateZone.bounds(1) + 6, immediateZone.bounds(3) + 1, applyHeightConstraints(heightConstraints);
                immediateZone.bounds(1) + 9, immediateZone.bounds(3) + 1, applyHeightConstraints(heightConstraints)
            ];
            
            upArm = [
                immediateZone.bounds(1) + 1, immediateZone.bounds(3) + 3, applyHeightConstraints(heightConstraints);
                immediateZone.bounds(1) + 1, immediateZone.bounds(3) + 6, applyHeightConstraints(heightConstraints);
                immediateZone.bounds(1) + 1, immediateZone.bounds(3) + 9, applyHeightConstraints(heightConstraints)
            ];
            
            lPositions = [rightArm; upArm];
            
        case 'bottom-right'
            leftArm = [
                immediateZone.bounds(2) - 3, immediateZone.bounds(3) + 1, applyHeightConstraints(heightConstraints);
                immediateZone.bounds(2) - 6, immediateZone.bounds(3) + 1, applyHeightConstraints(heightConstraints);
                immediateZone.bounds(2) - 9, immediateZone.bounds(3) + 1, applyHeightConstraints(heightConstraints)
            ];
            
            upArm = [
                immediateZone.bounds(2) - 1, immediateZone.bounds(3) + 3, applyHeightConstraints(heightConstraints);
                immediateZone.bounds(2) - 1, immediateZone.bounds(3) + 6, applyHeightConstraints(heightConstraints);
                immediateZone.bounds(2) - 1, immediateZone.bounds(3) + 9, applyHeightConstraints(heightConstraints)
            ];
            
            lPositions = [leftArm; upArm];
            
        case 'top-left'
            rightArm = [
                immediateZone.bounds(1) + 3, immediateZone.bounds(4) - 1, applyHeightConstraints(heightConstraints);
                immediateZone.bounds(1) + 6, immediateZone.bounds(4) - 1, applyHeightConstraints(heightConstraints);
                immediateZone.bounds(1) + 9, immediateZone.bounds(4) - 1, applyHeightConstraints(heightConstraints)
            ];
            
            downArm = [
                immediateZone.bounds(1) + 1, immediateZone.bounds(4) - 3, applyHeightConstraints(heightConstraints);
                immediateZone.bounds(1) + 1, immediateZone.bounds(4) - 6, applyHeightConstraints(heightConstraints);
                immediateZone.bounds(1) + 1, immediateZone.bounds(4) - 9, applyHeightConstraints(heightConstraints)
            ];
            
            lPositions = [rightArm; downArm];
            
        case 'top-right'
            leftArm = [
                immediateZone.bounds(2) - 3, immediateZone.bounds(4) - 1, applyHeightConstraints(heightConstraints);
                immediateZone.bounds(2) - 6, immediateZone.bounds(4) - 1, applyHeightConstraints(heightConstraints);
                immediateZone.bounds(2) - 9, immediateZone.bounds(4) - 1, applyHeightConstraints(heightConstraints)
            ];
            
            downArm = [
                immediateZone.bounds(2) - 1, immediateZone.bounds(4) - 3, applyHeightConstraints(heightConstraints);
                immediateZone.bounds(2) - 1, immediateZone.bounds(4) - 6, applyHeightConstraints(heightConstraints);
                immediateZone.bounds(2) - 1, immediateZone.bounds(4) - 9, applyHeightConstraints(heightConstraints)
            ];
            
            lPositions = [leftArm; downArm];
    end
    
    % Filter positions within bounds
    validLPositions = [];
    for i = 1:size(lPositions, 1)
        pos = lPositions(i, :);
        if pos(1) >= immediateZone.bounds(1) && pos(1) <= immediateZone.bounds(2) && ...
           pos(2) >= immediateZone.bounds(3) && pos(2) <= immediateZone.bounds(4)
            validLPositions = [validLPositions; pos];
        end
    end
    
    % Select best L-shape positions
    if size(validLPositions, 1) > lShapeTx
        positions = selectBestSpacedPositions(validLPositions, lShapeTx);
    else
        positions = validLPositions;
    end
    
    fprintf('L-shape placement: %d positions created\n', size(positions, 1));
end

function positions = generatePositionsInZone(zone, numPos, existingPositions, heightConstraints, minSeparation)
    %GENERATEPOSITIONSINZONE Generate valid positions within a zone
    
    positions = [];
    maxAttempts = 50;
    
    for i = 1:numPos
        position = [];
        attempts = 0;
        
        while isempty(position) && attempts < maxAttempts
            attempts = attempts + 1;
            
            % Generate candidate position based on zone type
            switch zone.type
                case 'circular'
                    candidate = generateCircularPosition(zone);
                case 'ring'
                    candidate = generateRingPosition(zone);
                case 'farField'
                    candidate = generateFarFieldPosition(zone);
                otherwise
                    candidate = generateRectangularPosition(zone);
            end
            
            % Add height
            candidate = [candidate, applyHeightConstraints(heightConstraints)];
            
            % Check if position is valid
            if isValidPosition(candidate, existingPositions, positions, minSeparation, zone)
                position = candidate;
            end
        end
        
        if ~isempty(position)
            positions = [positions; position];
        else
            fprintf('Warning: Could not place transmitter %d in %s zone\n', i, zone.name);
        end
    end
end

function pos = generateCircularPosition(zone)
    %GENERATECIRCULARPOSITION Generate position within circular zone
    
    angle = rand * 2 * pi;
    radius = rand * zone.radius;
    
    x = zone.center(1) + radius * cos(angle);
    y = zone.center(2) + radius * sin(angle);
    
    % Clamp to zone bounds
    x = max(zone.bounds(1), min(zone.bounds(2), x));
    y = max(zone.bounds(3), min(zone.bounds(4), y));
    
    pos = [x, y];
end

function pos = generateRingPosition(zone)
    %GENERATERINGPOSITION Generate position within ring zone
    
    angle = rand * 2 * pi;
    radius = zone.innerRadius + rand * (zone.outerRadius - zone.innerRadius);
    
    x = zone.center(1) + radius * cos(angle);
    y = zone.center(2) + radius * sin(angle);
    
    % Clamp to zone bounds
    x = max(zone.bounds(1), min(zone.bounds(2), x));
    y = max(zone.bounds(3), min(zone.bounds(4), y));
    
    pos = [x, y];
end

function pos = generateFarFieldPosition(zone)
    %GENERATEFARFIELDPOSITION Generate position in far field zone
    
    maxAttempts = 20;
    
    for attempt = 1:maxAttempts
        x = zone.bounds(1) + rand * (zone.bounds(2) - zone.bounds(1));
        y = zone.bounds(3) + rand * (zone.bounds(4) - zone.bounds(3));
        
        % Check if outside exclusion radius
        dist = norm([x, y] - zone.center);
        if dist > zone.exclusionRadius
            pos = [x, y];
            return;
        end
    end
    
    % Fallback to edge of exclusion zone
    angle = rand * 2 * pi;
    x = zone.center(1) + (zone.exclusionRadius + 2) * cos(angle);
    y = zone.center(2) + (zone.exclusionRadius + 2) * sin(angle);
    
    % Clamp to zone bounds
    x = max(zone.bounds(1), min(zone.bounds(2), x));
    y = max(zone.bounds(3), min(zone.bounds(4), y));
    
    pos = [x, y];
end

function pos = generateRectangularPosition(zone)
    %GENERATERECTANGULARPOSITION Generate position within rectangular zone
    
    x = zone.bounds(1) + rand * (zone.bounds(2) - zone.bounds(1));
    y = zone.bounds(3) + rand * (zone.bounds(4) - zone.bounds(3));
    pos = [x, y];
end

function valid = isValidPosition(candidate, existingPositions, currentPositions, minSeparation, zone)
    %ISVALIDPOSITION Check if position meets all constraints
    
    valid = true;
    
    % Check bounds
    if candidate(1) < zone.bounds(1) || candidate(1) > zone.bounds(2) || ...
       candidate(2) < zone.bounds(3) || candidate(2) > zone.bounds(4)
        valid = false;
        return;
    end
    
    % Check separation from existing positions
    if ~isempty(existingPositions)
        distances = sqrt(sum((existingPositions(:, 1:2) - candidate(1:2)).^2, 2));
        if any(distances < minSeparation)
            valid = false;
            return;
        end
    end
    
    % Check separation from current positions
    if ~isempty(currentPositions)
        distances = sqrt(sum((currentPositions(:, 1:2) - candidate(1:2)).^2, 2));
        if any(distances < minSeparation)
            valid = false;
            return;
        end
    end
    
    % Additional zone-specific constraints
    if strcmp(zone.type, 'ring')
        dist = norm(candidate(1:2) - zone.center);
        if dist < zone.innerRadius || dist > zone.outerRadius
            valid = false;
        end
    elseif strcmp(zone.type, 'farField')
        dist = norm(candidate(1:2) - zone.center);
        if dist < zone.exclusionRadius
            valid = false;
        end
    end
end

function height = applyHeightConstraints(heightConstraints)
    %APPLYHEIGHTCONSTRAINTS Apply height constraints with variation
    
    baseHeight = heightConstraints.preferredHeight;
    variation = heightConstraints.heightVariation * (2*rand - 1);
    height = baseHeight + variation;
    
    height = max(heightConstraints.minHeight, height);
    height = min(heightConstraints.maxHeight, height);
end

function positions = fillWithDistancePriority(zones, existingPositions, numPos, heightConstraints, minSeparation)
    %FILLWITHDISTANCEPRIORITY Fill remaining positions with distance priority
    
    positions = [];
    
    % Priority order: immediate -> nearField -> oppositeDiagonal -> farField
    zoneOrder = {'immediate', 'nearField', 'oppositeDiagonal', 'farField'};
    
    for i = 1:length(zoneOrder)
        zoneName = zoneOrder{i};
        if ~isfield(zones, zoneName) || size(positions, 1) >= numPos
            continue;
        end
        
        zone = zones.(zoneName);
        remainingPos = numPos - size(positions, 1);
        
        allExisting = [existingPositions; positions];
        zonePositions = generatePositionsInZone(zone, remainingPos, allExisting, heightConstraints, minSeparation);
        positions = [positions; zonePositions];
        
        if size(positions, 1) >= numPos
            break;
        end
    end
    
    % Trim to exact number if needed
    if size(positions, 1) > numPos
        positions = positions(1:numPos, :);
    end
end

function combinedPositions = addValidPositions(existingPositions, newPositions, minSeparation)
    %ADDVALIDPOSITIONS Add new positions that meet separation requirements
    
    combinedPositions = existingPositions;
    
    for i = 1:size(newPositions, 1)
        candidate = newPositions(i, :);
        
        % Check separation from existing positions
        valid = true;
        if ~isempty(combinedPositions)
            distances = sqrt(sum((combinedPositions(:, 1:2) - candidate(1:2)).^2, 2));
            if any(distances < minSeparation)
                valid = false;
            end
        end
        
        if valid
            combinedPositions = [combinedPositions; candidate];
        end
    end
end

function selectedPositions = selectBestSpacedPositions(positions, numSelect)
    %SELECTBESTSPACEDPOSITIONS Select well-spaced positions using greedy algorithm
    
    if size(positions, 1) <= numSelect
        selectedPositions = positions;
        return;
    end
    
    % Greedy algorithm to select well-spaced positions
    selectedPositions = positions(1, :); % Start with first position
    remaining = positions(2:end, :);
    
    while size(selectedPositions, 1) < numSelect && ~isempty(remaining)
        % Find position with maximum minimum distance to selected positions
        bestIdx = 1;
        bestMinDist = 0;
        
        for i = 1:size(remaining, 1)
            candidate = remaining(i, :);
            distances = sqrt(sum((selectedPositions(:, 1:2) - candidate(1:2)).^2, 2));
            minDist = min(distances);
            
            if minDist > bestMinDist
                bestMinDist = minDist;
                bestIdx = i;
            end
        end
        
        selectedPositions = [selectedPositions; remaining(bestIdx, :)];
        remaining(bestIdx, :) = [];
    end
end

function optimizedPositions = optimizeGeometry(txPositions, rxPosition, roomBounds, heightConstraints, minSeparation)
    %OPTIMIZEGEOMETRY Optimize transmitter geometry for GDOP minimization
    
    fprintf('Optimizing geometry for GDOP minimization...\n');
    
    optimizedPositions = txPositions;
    initialGDOP = calculateGDOP(txPositions, rxPosition);
    
    fprintf('Initial GDOP: %.2f\n', initialGDOP);
    
    % Iterative improvement
    maxIterations = 5;
    stepSize = 1.0; % 1 meter adjustment steps
    
    for iter = 1:maxIterations
        improved = false;
        
        for i = 1:size(optimizedPositions, 1)
            currentGDOP = calculateGDOP(optimizedPositions, rxPosition);
            originalPos = optimizedPositions(i, :);
            
            % Try small adjustments
            adjustments = [
                [stepSize, 0, 0]; [-stepSize, 0, 0];
                [0, stepSize, 0]; [0, -stepSize, 0];
                [stepSize, stepSize, 0]; [-stepSize, -stepSize, 0];
            ];
            
            for j = 1:size(adjustments, 1)
                testPos = originalPos + adjustments(j, :);
                
                % Check if adjustment is valid
                if isAdjustmentValid(testPos, optimizedPositions, i, roomBounds, minSeparation)
                    % Test GDOP improvement
                    testPositions = optimizedPositions;
                    testPositions(i, :) = testPos;
                    testGDOP = calculateGDOP(testPositions, rxPosition);
                    
                    if testGDOP < currentGDOP
                        optimizedPositions(i, :) = testPos;
                        improved = true;
                        break;
                    end
                end
            end
        end
        
        if ~improved
            break;
        end
    end
    
    finalGDOP = calculateGDOP(optimizedPositions, rxPosition);
    improvement = (initialGDOP - finalGDOP) / initialGDOP * 100;
    
    fprintf('Geometry optimization: GDOP %.2f -> %.2f (%.1f%% improvement)\\n', ...
        initialGDOP, finalGDOP, improvement);
end

function valid = isAdjustmentValid(testPos, positions, excludeIdx, roomBounds, minSeparation)
    %ISADJUSTMENTVALID Check if position adjustment is valid
    
    valid = true;
    
    % Check room bounds
    if testPos(1) < roomBounds.xmin || testPos(1) > roomBounds.xmax || ...
       testPos(2) < roomBounds.ymin || testPos(2) > roomBounds.ymax
        valid = false;
        return;
    end
    
    % Check separation from other transmitters
    otherPositions = positions([1:excludeIdx-1, excludeIdx+1:end], :);
    if ~isempty(otherPositions)
        distances = sqrt(sum((otherPositions(:, 1:2) - testPos(1:2)).^2, 2));
        if any(distances < minSeparation)
            valid = false;
        end
    end
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
    
    angles = [];
    for i = 1:size(txPositions, 1)
        angle = atan2d(txPositions(i, 2) - rxPosition(2), txPositions(i, 1) - rxPosition(1));
        angles = [angles; angle];
    end
    
    angles = sort(angles);
    angleDiffs = diff([angles; angles(1) + 360]);
    avgAngle = mean(angleDiffs);
end

function validatePlacement(txPositions, rxPosition, roomBounds, nearFieldRadius)
    %VALIDATEPLACEMENT Validate and analyze transmitter placement
    
    fprintf('\n=== Placement Validation ===\n');
    
    numTx = size(txPositions, 1);
    
    % Calculate distances from receiver to each transmitter (using 3D norm)
    distances = zeros(numTx, 1);
    for i = 1:numTx
        distances(i) = norm(txPositions(i, :) - rxPosition);
    end
    
    % Distance analysis
    minDist = min(distances);
    maxDist = max(distances);
    avgDist = mean(distances);
    
    fprintf('Distance analysis (3D):\n');
    fprintf('  Minimum distance: %.1fm\n', minDist);
    fprintf('  Maximum distance: %.1fm\n', maxDist);
    fprintf('  Average distance: %.1fm\n', avgDist);
    
    % Check near field coverage
    nearFieldTx = sum(distances <= nearFieldRadius);
    fprintf('  Transmitters within %.1fm: %d/%d (%.1f%%)\n', ...
        nearFieldRadius, nearFieldTx, numTx, nearFieldTx/numTx*100);
    
    % Calculate GDOP
    gdop = calculateGDOP(txPositions, rxPosition);
    fprintf('  Final GDOP: %.2f\n', gdop);
    
    % Angular diversity
    avgAngularSep = calculateAverageAngularSeparation(txPositions, rxPosition);
    
    fprintf('  Average angular separation: %.1f degrees\n', avgAngularSep);
    
    % Performance assessment
    fprintf('\nPerformance Assessment:\n');
    
    if minDist < 15
        fprintf('  Good near-field coverage (closest Tx at %.1fm)\n', minDist);
    else
        fprintf('  Poor near-field coverage (closest Tx at %.1fm)\n', minDist);
    end
    
    if gdop < 5
        fprintf('  Excellent GDOP (%.2f)\n', gdop);
    else
        fprintf('  Moderate GDOP (%.2f)\n', gdop);
    end
    
    if avgAngularSep > 45
        fprintf('  Good angular diversity (%.1f degrees)\n', avgAngularSep);
    else
        fprintf('  Poor angular diversity (%.1f degrees)\n', avgAngularSep);
    end
end
