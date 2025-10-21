function [viewer, modelBounds, roomDims] = setupEnvironment(config)
%SETUPENVIRONMENT Load the 3D model and create a siteviewer
% Inputs:
%   config      - PositioningConfig with environment settings
% Outputs:
%   viewer      - siteviewer handle for 3D visualization
%   modelBounds - struct with fields xmin/xmax/ymin/ymax/zmin/zmax (m)
%   roomDims    - [L W H] dimensions (m)
    
    fprintf('Setting up 3D environment...\n');
    
    % Load 3D model with scaling
    if config.useCustomModel && exist(config.customModelFile, 'file')
        try
            % Load custom model with scaling
            viewer = siteviewer('SceneModel', config.customModelFile, ...
                               'SceneModelScale', config.modelScaleFactor, ...
                               'Name', '5G Positioning System - Large Indoor Environment');
            fprintf('Loaded custom model: %s (scaled by %.1fx)\n', ...
                config.customModelFile, config.modelScaleFactor);
            
            % Extract model dimensions for automatic transmitter placement
            [roomDims, modelBounds] = extractModelDimensions(config.customModelFile, ...
                                                           config.modelScaleFactor);
            
        catch ME
            fprintf('Failed to load custom model: %s\n', ME.message);
            fprintf('Using built-in conference room model\n');
            [viewer, modelBounds, roomDims] = setupBuiltInModel();
        end
    else
        [viewer, modelBounds, roomDims] = setupBuiltInModel();
    end
    
    % Set viewing properties
    viewer.Basemap = 'none';
    
    fprintf('Environment setup complete: %.1f x %.1f x %.1f meters\n', roomDims);
end

function [viewer, modelBounds, roomDims] = setupBuiltInModel()
    %SETUPBUILTINMODEL Setup built-in conference room model
    
    try
        % Try to load built-in conference room model
        viewer = siteviewer('SceneModel','conferenceroom.stl', ...
                           'Name', '5G Positioning System - Conference Room');
        fprintf('Using built-in conference room model\n');
        roomDims = [10, 8, 3]; % Default dimensions
        modelBounds = struct('xmin', -5, 'xmax', 5, 'ymin', -4, 'ymax', 4, 'zmin', 0, 'zmax', 3);
    catch
        % If no built-in model, create basic indoor environment
        viewer = siteviewer('Basemap','none', ...
                           'Name', '5G Positioning System - Basic Environment');
        fprintf('Using basic indoor environment\n');
        roomDims = [10, 8, 3]; % Default dimensions
        modelBounds = struct('xmin', -5, 'xmax', 5, 'ymin', -4, 'ymax', 4, 'zmin', 0, 'zmax', 3);
    end
end

function [dimensions, bounds] = extractModelDimensions(modelFile, scaleFactor)
    %EXTRACTMODELDIMENSIONS Extract dimensions from 3D model file
    
    try
        if contains(modelFile, '.stl', 'IgnoreCase', true)
            % Read STL file
            [vertices, ~] = readSTL(modelFile);
            
            % Apply scale factor
            vertices = vertices * scaleFactor;
            
            % Calculate bounds
            bounds.xmin = min(vertices(:,1));
            bounds.xmax = max(vertices(:,1));
            bounds.ymin = min(vertices(:,2));
            bounds.ymax = max(vertices(:,2));
            bounds.zmin = min(vertices(:,3));
            bounds.zmax = max(vertices(:,3));
            
        elseif contains(modelFile, '.gltf', 'IgnoreCase', true)
            % For GLTF files, use approximate dimensions
            bounds.xmin = -5 * scaleFactor;
            bounds.xmax = 5 * scaleFactor;
            bounds.ymin = -5 * scaleFactor;
            bounds.ymax = 5 * scaleFactor;
            bounds.zmin = 0;
            bounds.zmax = 3 * scaleFactor;
        else
            error('Unsupported file format');
        end
        
        % Calculate dimensions
        dimensions = [
            bounds.xmax - bounds.xmin;  
            bounds.ymax - bounds.ymin;  
            bounds.zmax - bounds.zmin   
        ];
        
        fprintf('Model bounds: X[%.1f,%.1f] Y[%.1f,%.1f] Z[%.1f,%.1f]\n', ...
            bounds.xmin, bounds.xmax, bounds.ymin, bounds.ymax, bounds.zmin, bounds.zmax);
        
    catch ME
        fprintf('Failed to extract model dimensions: %s\n', ME.message);
        % Use default dimensions
        dimensions = [10, 8, 3];
        bounds.xmin = -5; bounds.xmax = 5;
        bounds.ymin = -4; bounds.ymax = 4;
        bounds.zmin = 0; bounds.zmax = 3;
    end
end

function [vertices, faces] = readSTL(filename)
    %READSTL Simple STL reader for extracting vertices
    
    fid = fopen(filename, 'r');
    if fid == -1
        error('Cannot open STL file: %s', filename);
    end
    
    % Check if binary or ASCII
    header = fread(fid, 80, 'char');
    if ~isempty(strfind(char(header'), 'solid'))
        % ASCII format
        fclose(fid);
        [vertices, faces] = readSTL_ASCII(filename);
    else
        % Binary format
        fclose(fid);
        [vertices, faces] = readSTL_Binary(filename);
    end
end

function [vertices, faces] = readSTL_ASCII(filename)
    %READSTL_ASCII Read ASCII STL file
    
    fid = fopen(filename, 'r');
    vertices = [];
    faces = [];
    faceCount = 0;
    
    while ~feof(fid)
        line = fgetl(fid);
        if contains(line, 'vertex')
            coords = sscanf(line, ' vertex %f %f %f');
            if length(coords) == 3
                vertices = [vertices; coords'];
            end
        elseif contains(line, 'endfacet')
            faceCount = faceCount + 1;
            if size(vertices, 1) >= faceCount * 3
                startIdx = (faceCount - 1) * 3 + 1;
                faces = [faces; startIdx, startIdx+1, startIdx+2];
            end
        end
    end
    fclose(fid);
end

function [vertices, faces] = readSTL_Binary(filename)
    %READSTL_BINARY Read binary STL file
    
    fid = fopen(filename, 'r');
    fread(fid, 80, 'char'); % Skip header
    numTriangles = fread(fid, 1, 'uint32');
    
    vertices = zeros(numTriangles * 3, 3);
    faces = zeros(numTriangles, 3);
    
    for i = 1:numTriangles
        fread(fid, 3, 'float32'); % Skip normal vector
        
        % Read 3 vertices
        for j = 1:3
            vertex = fread(fid, 3, 'float32');
            vertexIdx = (i-1) * 3 + j;
            vertices(vertexIdx, :) = vertex';
        end
        
        fread(fid, 1, 'uint16'); % Skip attribute byte count
        
        % Define face
        startIdx = (i-1) * 3 + 1;
        faces(i, :) = [startIdx, startIdx+1, startIdx+2];
    end
    
    fclose(fid);
end
