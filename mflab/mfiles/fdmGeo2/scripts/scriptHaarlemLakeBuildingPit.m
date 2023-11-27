%% Haarlemmermeer building pit

clear;
close all

%% data
kD = 1500;
c  =  600;
S  = 0.001;
lambda = sqrt(1500*600);
hp =   -6;  % NAP surface water Haarlemmermeer
hl =   -1;  % NAP surface water outside HLM
hBp=   -9;  % NAP dewatering level below building pit
rw =  0.25; % well radius
t=2;

% extraction wells
exWells.x = [ -20 -20 -20   0  20 20 20  0 ] - 450;
exWells.y = [  20   0 -20 -20 -20  0 20 20 ] -   0;
nEx       = numel(exWells.x);

% injection wells
inWells.x = (0:-20:-460) - 300;
inWells.y = (0:+20:+460) + 500;
nIn       = numel(inWells.x);

%% analytical estimate
phi= hp + (0.5*(hl+hp)-hp) * exp(mean(exWells.x)/lambda);
dd = phi - hBp;
QQ = 11000;
r  = 20;
s  = QQ/(2*pi*kD) * besselk(0,r/lambda)/(r/lambda*besselk(1,r/lambda));

%% extraction wells around building pit
Q = -QQ/nEx;
exWells.Q = ones(1,nEx) * Q;

%% injection wells
inWells.Q = -Q * nEx/nIn * ones(1,nIn);

%% grid
x = unique([ -1000:100:1000 -520:10:400  min(inWells.x)-60:20:max(inWells.x)+60]);
y = unique([  -500:100:1500  -60:10:60   min(inWells.y)-60:20:max(inWells.y)+60])';
y = y(end:-1:1);

X = ones(size(y)) * x;
Y = y * ones(size(x));

%% initial head (polder formula)
phi      = zeros(size(X));
phi(X<=0)= hp + (0.5*(hl+hp)-hp) * exp(+X(X<=0)/lambda);
phi(X> 0)= hl + (0.5*(hl+hp)-hl) * exp(-X(X> 0)/lambda);

%% And now: superposition !
Phi = phi; % initial head

% superimpose extraction wells
fprintf('Superimposing %2d exWells:',nEx);
for iw=1:nEx
    R = sqrt((exWells.x(iw)-X).^2+(exWells.y(iw)-Y).^2); R(R<rw)=rw;
    rho = R/lambda;
    u = (R.^2 * S)/(4*kD*t);
    Phi = Phi + exWells.Q(iw)/(4*pi*kD) * Wh(u,rho);
    fprintf(' %d',iw);
end
fprintf('\n');

% superimpose injection wells
fprintf('Superimposing %2d inWells:',nIn);
for iw=1:nIn
    R = sqrt((inWells.x(iw)-X).^2+(inWells.y(iw)-Y).^2); R(R<rw)=rw;
    rho = R/lambda;
    u = (R.^2 * S)/(4*kD*t);
    Phi = Phi + inWells.Q(iw)/(4*pi*kD) * Wh(u,rho);
    fprintf(' %d',iw);
end
fprintf('\n');

%% show results
figure; hold on; xlabel('x [m]'); ylabel('y [m]');
title(sprintf('Head HLmeer during extraction and injection (t=%.0f d)',t))

hrange = floor(min(Phi(:))):0.25:ceil(max(Phi(:)));
contourf(X,Y,Phi,hrange);
plot(exWells.x,exWells.y,'yo');
plot(inWells.x,inWells.y,'co');

hb = colorbar; set(get(hb,'title'),'string','head NAP');


%% show results
figure; hold on; xlabel('x [m]'); ylabel('y [m]');
title(sprintf('Head HLmeer above ground surface (t=%.0f d)',t))

DEM = zeros(size(X)) -4.5; DEM(X>0) = 0;

hrange = floor(min(Phi(:)-DEM(:))):0.2:ceil(max(Phi(:)-DEM(:)));
hcrit  = [0.5 0.5]; % critical values for breaking peat layer

contourf(X,Y,Phi-DEM,hrange);
contour (X,Y,Phi-DEM,hcrit,'w','lineWidth',2);
plot(exWells.x,exWells.y,'yo');
plot(inWells.x,inWells.y,'co');

hb = colorbar; set(get(hb,'title'),'string','head-DEM');
