function [e, n, u] = SRTM_CalculateLocalTangentFrame(HeightArray, Grid, LatitudeReference, LongitudeReference, HeightReference)
% SRTM_CalculateLocalTangentFrame
%
% Calculates each point a given grid in the local tangent frame for a given
% reference point

% Get the size of the array.
ArraySize = size(HeightArray);

% preallocate the east, north and up frame
e = zeros(ArraySize);
n = zeros(ArraySize);
u = zeros(ArraySize);

% Put the latitudes, longitudes, grid, etc. in RADIANS
Grid = Grid * (pi/180);
LatitudeReference = LatitudeReference * (pi/180);
LongitudeReference = LongitudeReference * (pi/180);

% Perform the operation on a column-by-column basis
for i = 1:ArraySize(1)
    
    % Convert the points into the local frame
    [e(i,:),n(i,:),u(i,:)] = CoordinateTransform_LLHtoENU( ...
        LatitudeReference, LongitudeReference, HeightReference, ...
        Grid(i,:,1), Grid(i,:,2), HeightArray(i,:) );
end

