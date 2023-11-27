% Scripts shows use of drilling database boring.mat
% TO 140509

F = '~/GRWMODELS/AWD/AWD-data/Boringen/boring.mat';
load(F); % load dabase boring.mat

% NSea coast coordinates to compute distance from well to coast
NSea.lon = [ 4.49393426895234, 4.51654442259011]';
NSea.lat = [52.32963844839084,52.36426877380028]';
[NSea.x,NSea.y] = wgs2rd(NSea.lon,NSea.lat);

%% Wells in profiles cc Stufzand (1988) as far as available in boring.mat
cc = {
'24H189'
'24H301'
'24H190'
'24H425'
'24H717'
'24H070'
'24H018'
'24H428'
'24H371'
'25C008'
'25C331'
'25C332'
'25C012'
'25C181'
};

% get them
B = boringObj(cc,boring);

% show piezometers along profile throught drillings
figure; hold on; title('bore holes vs distance along profile');
xlabel('x [m]'); ylabel('z [m]');
B.plot;

% show piezometrs versus distance to North Sea coast
figure; hold on; title('bore holes vs distance from North Sea coast');
xlabel('x [m]'); ylabel('z[m]');
B.plot(NSea.x,NSea.y);

