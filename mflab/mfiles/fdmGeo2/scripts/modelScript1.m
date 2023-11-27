%%% modelScript -- sets up 2D FDM, runs it and shows results

FIXED = -1;   % indicatre fixed head node in IBOUND

%% Grid
d   = 25; % cell width, also try 5 m
xGr = [-1000:d:-d -d/2 d/2 d:d:1000];
yGr = [-1000:d:-d -d/2 d/2 d:d:1000];
gr  =  grid2DObj(xGr,yGr);

%% Island
R0     =  800; % +d/2;
R      = sqrt(gr.Xm.^2 + gr.Ym.^2);
IBOUND = ones(gr.size);
IBOUND(R>=R0) = FIXED;

%% Transmissivities
Tx = 600; Ty = 600;

%% Start heads, initial heads
IH = gr.const(0);

%% Fixed inflows of cells
rch = 0.001; % net recharge rate
FQ  = gr.Area * rch;

%% Run model
[Phi,Q,Qx,Qy] = fdm2(gr,Tx,Ty,IBOUND,IH,FQ);

%% Visualize results
figure; hold on;
xlabel('x [m]'); ylabel('y [m]');
title('Heads in island');

phiMax = max(Phi(:));  hRange = phiMax*(0:25)/25;
contourf(gr.xm,gr.ym,Phi,hRange,'edgeColor','none');

hb=colorbar; set(get(hb,'title'),'string','head [m]');

%% Analytical comparison

% Analytical solution for head in island
PhiA = rch/4./Tx .* (R0.^2 - R.^2);

% black contours over the colored contours
contour(gr.xm,gr.ym,PhiA,hRange,'k')

%% Plot X-section to compare with analytical results
figure; hold on;
xlabel('x [m]'); ylabel('head [m]');
title('head through center of island');

Iy = find(gr.ym==0);

plot(gr.xm,Phi (Iy,:),'b');
plot(gr.xm,PhiA(Iy,:),'r.');

%% Check water balance
fprintf('Water balances:\n');
fprintf('Total water balance = %10g (should be zero)\n',sum(Q(:)));
fprintf('Total recharge      = %10.0f m3/d (should be pi*R0^2*rch = %6.0f m3/d)\n',sum(Q(Q>0)),pi*R0^2*rch);
fprintf('Total dicharge      = %10.0f m3/d (should match with total recharge.\n',sum(Q(Q<0)));

