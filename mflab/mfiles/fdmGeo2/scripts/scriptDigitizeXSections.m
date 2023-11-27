% scriptDigitizeXSections -- demonstration of digitizing directly from
% figure in Matab. This always works, no matter whether the figure contains
% graphs or an image.

%% To digitize
% Make sure you have a figure with an axis. Then make it active by clicking
% on it. Finally digitze using ginput

%[xx yy] = ginput()

% The xx and yy will have the digitized points. Return finishes the input
% Type 
%    doc ginput
% for further information on that function.

%% To digitize from an image
% If you laod and show an image like

%A = imread('figureName');

% and then show it using

%image(A)

%You will see the image with pixel axes.

% If however, you show the image with axies values given like

% image(x,y,A);

% then the image will be placed on these axes. So to put an image on its
% real world axes you only have to supply the extents or all values of
% those axes. The only problem is how to get these real world extents.

% The basic idea is georeferencing with two points in the figure that you
% know the real world coordinates of. Digitize them as above and the
% compute the real world extens of the entire image using its pixel size
% obtained from
% size(A)

% Using two anonumous functions you  can compute any real world coordinate
% from the digizized two points whos x and y coordinatea are xp and yp
% respectively, which correspond to world coordinates xW and yW
% respectively:

%x = @(x) xW(1) + (x-xW(1))*diff(xp)/diff(xW);
%y = @(y) yW(1) + (y-yW(1))*diff(yp)/diff(yW);

%xE = x([1 size(A,2)]);  % [1 size(A,2)] extent of x-axis in pixels
%yE = y([1 size(A,1)]);  % [1 size(A,1)] extent of y-axis in pixels 

% then reload the image with the axes

% image(xE,yE,A);
% 
% and start digitizing
% 
% [xw yw] = ginput();


%% Digitizing from an image using geoRef

% This above method is handwork. The method has been combined in a single
% functio

%xy = geoRef(figureName,P);

%where

% P = [xE(1) yE(1); xE(2) yE(2)];

% i.e. the coordinates of the upper left and lower right point of the figure.

% What it does?
% geoRef load the figure. You digitize the two known points of which you
% supplied the real world coordinates in P. After two points have been
% digitized the image is automatically renewed, but now shows it on its
% real world axes. You can immediately continue digitizing in the figure.
% Each mouse click adds a point. The point is then shown connected to the
% previouw point with a dashed line. You can remove points using backspace.
% You can continue with a new  line by pressing any key (preferabl the
% space bar). The new line will have a different color. And so on. When
% finished press the return key. The result is a cell array each element of
% which contains the coordinates of a digitized line. This way many lines
% can be digitized in a single sequence.
% Warning: you cannot backspace to a previous line. You can only remove
% previous points of the current line.

% Example: Try a google earh image. It may be very useful as background
% image to put model results on. You may use the function

% [x,y] = wgs2rd(lon,lat)

% to convert WGS84 coordinates to Rijksdriehoek Dutch system. If you do
% that for the two points in the image that you know the coordinates of,
% the entire image will be shown in RD coordinates, which are most
% convenient to work in The Netherlands.

% Example: Digitize in a cross section. The two points to be digized are
% the west most piezometer and the right-most piezometer which are 42 m
% apart. The elevations are aribitray as long as sufficiently distinct,
% they can be read off at the left axis in the original figure

% (The points to be digitized are the left most and right most piezometers
% at 42 m distance from each other. The depths can be taken from the vertical
% axis of the figure itself).

%xy  = geoRef('crosssectionP8D9U5+P12D8U3.tiff',[0 6; 42 -20]);

% Example 3, same situation.

% (The points to digitize are the left most and right most piezometers
% at 29.5 m mutual distance. The depths can be taken from the vertical axis
% of teh figure itself.

if ismac
    xy = geoRef('../duneData/CrossSec10J536-537.png',[0 4; 92 -16]);
else
    xy = geoRef('..\duneData\CrossSec10J536-537.png',[0 4; 92 -16]);
end
