 function [waveform, resourceGrid, info] = generateEnhancedPRSWaveform(carrier, prs, numSlots)
%GENERATEENHANCEDPRSWAVEFORM Generate PRS waveform with muting support
% Inputs:
%   carrier  - nrCarrierConfig
%   prs      - nrPRSConfig
%   numSlots - number of slots
% Outputs:
%   waveform     - complex waveform (column)
%   resourceGrid - resource grid used for modulation
%   info         - OFDM info struct
    
    try
        
        % Initialize enhanced resource grid
        info = nrOFDMInfo(carrier);
        resourceGrid = nrResourceGrid(carrier, numSlots);
        
        % Generate and map PRS symbols with muting consideration
        for slotIdx = 0:numSlots-1
            carrier.NSlot = slotIdx;
            % Rely on 5G Toolbox to handle slot mapping and muting
            prsSymbols = nrPRS(carrier, prs);
            prsIndices = nrPRSIndices(carrier, prs);
            % Map PRS to resource grid (will be empty in muted/absent slots)
            if ~isempty(prsIndices) && ~isempty(prsSymbols)
                resourceGrid(prsIndices) = prsSymbols;
            end
        end
        
        % Generate OFDM waveform
        waveform = nrOFDMModulate(carrier, resourceGrid);
        
        % Ensure column vector
        if size(waveform, 2) > 1
            waveform = waveform(:);
        end
        
        fprintf('Enhanced PRS waveform generated: %dx%d samples\n', size(waveform));
        
    catch ME
        fprintf('Enhanced PRS waveform generation failed: %s\n', ME.message);
        error('Failed to generate enhanced PRS waveform');
    end
end
