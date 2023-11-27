%%% ModelScript -- Test random walk when Diff not constant
%   Runs a particle tracking model (Moc method of characteristics)
%   For case as in Uffink (1990)
%   Shows the results
%   TO 140427

%% Cleanup
close all;

%% Constants used in IBOUND
FXHD     = -1;
INACTIVE =  0;
ACTIVE   =  1;

clr = 'brgkmcy';  % list of colors

%% Generate the Grid for th FDM flow modoel
xGr   = -20:4:20;
yGr   = [-20 20];   % 5000;
zGr   = [0 -1];
gr    =  grid2DObj(xGr,yGr,zGr); % generates 2D gridObj with depth

% Show the grid
% figure; hold on;
% xlabel('x [m]'); ylabel('y [m]');
% title('example2: MOC, flow from left to right');
% gr.plot('c');  % lines in cyan color

%% Transmissivities
Tx = gr.const(1);
Ty = gr.const(1);

%% Recharge
rch   = 0.0;   % 01; % net recharge rate
FQ    = gr.Area * rch;

IH  = gr.const(0); IH(:,1)=20;

%% IBOUND (specifies where heads are fixed)
IBOUND = ones(gr.size);
IBOUND(:,1) = FXHD;


%% Run the flow model
[Phi,Q,Qx,Qy] = fdm2(gr,Tx,Ty,IBOUND,IH,FQ);

%% Setup the particle tracking model
t    = 0:100:5000; % times
%t    = 0:1:100;
Peff = gr.const(0.35); % effective porosity

aL   = 1;   % [ m ] longitudinal dispersivity
aT   = aL/10; % [ m ] transversal dispersivity
Diff = 1e-4;  % [m2/d] diffusion coefficient
R    = 1;     % [ - ] retardation
lambda = 2e-5;% [1/d] decay

%% Generate starting particles and run MOC
n = -30;

D0   = 1e-3;
Diff = D0*(0.25 + 2.5 * abs(gr.xm)/abs(gr.xm(end)));

P = Moc(gr,Qx,Qy,Peff,R,t,n,'aL',aL,'aT',aT,'Diff',Diff); %,'lambda',lambda);

%% Visualize results
close all;

figure();
set(gca,'nextplot','add','xlim',gr.xGr([1 end]),'ylim',gr.yGr([end 1]));

xlabel('x [m]'); ylabel('y [m]');
title('example2: MOC, flow from left to right');

% Contour heads
phiMax = max(Phi(:)); phiMin = min(Phi(:));  hRange = phiMin:(phiMax-phiMin)/25:phiMax;
try
    contourf(gr.xm,gr.ym,Phi,hRange,'edgeColor','none');
catch %#ok
    contourf(gr.xm,gr.yGr,[Phi;Phi],hRange,'edgeColor','none');
end

gr.plot('c'); % Plot grid on top

% Show moving particles (stored in struct P)
time = [P.time];
Np   = numel(P(1).x);
for it=1:numel(time)
    % Title will change according to passing time
    ttl = sprintf('Tracking %d particles time = %.0f d',Np,time(it));
    if it==1
        % First loop, title and plot
        ht = title(ttl);
        h = plot(P(it).x,P(it).y,'k.','markerSize',3);
    else
        % Subsequent loops, reset title and points
        set(ht,'string',ttl);
        set(h,'xData',P(it).x,'yData',P(it).y);
        drawnow();  % necessary to update plot
        pause(0.1); % smooth movie
    end
end
    
%% Plot number of particles in each cell
figure; hold on; grid on;
ht = title(sprintf('Nr of particles in cells, total simulated = %d',Np));
xlabel('time'); ylabel('total nr of particles');

% Make array of cell nrs in which particles are after each time step
PIcell = [P.Icells];

N=[]; N(numel(time),gr.Nod)=0;
for it=1:numel(time)
    for ic=1:gr.Nod
        N(it,ic) = sum(P(it).Icells==ic);
    end
    fprintf('.');
end
fprintf('\n');

set(gca,'ylim',[0 n*n*1.5]);
for it=1:numel(time)
    plot(1:gr.Nod,N(it,:));
end

%% Plot arrival times of particles at different distancees
figure; hold on; grid on;
ht = title(sprintf('Nr of particles in cells, total simulated = %d',Np));
xlabel('time'); ylabel('total nr of particles');


PX = [P.x];

N = zeros(numel(time),gr.Nx);
for it=1:numel(time)    
    for ix=1:gr.Nx
        N(:,ix) = sum(sum(PX>gr.xm(ix)));
    end
end

for ix = 1:gr.Nx
    plot(time,N(:,ix));
end


%% Check water balance
fprintf('Water balances:\n');
fprintf('Total water balance = %10g (should be zero)\n',sum(Q(IBOUND~=0)));
fprintf('Total recharge (active  cells) = %10.0f m3/d\n',sum(Q(IBOUND>0)));
fprintf('Total discharge(fixhd + wells) = %10.0f m3/d\n',sum(Q(IBOUND<0)));

