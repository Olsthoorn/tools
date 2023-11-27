%%% ModelScript -- Test random walk column experiment Maas
%   Runs a particle tracking model (Moc method of characteristics)
%   For case as in Maas (1996)
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
xGr   = 0:100:15000; xGr = [xGr(1)-diff(xGr([1 2])) xGr xGr(end)+100];
yGr   = 0:100:1000;   % 5000;
zGr   = [0 -40];
gr    =  grid2DObj(xGr,yGr,zGr); % generates 2D gridObj with depth

Peff   = gr.const(0.33); % effective porosity

%% Transmissivities
mT = 25;
vT = 50^2;
mu = log(mT^2/sqrt(vT+mT^2)); v  =log(vT/mT^2+1);

Tx = reshape(exp(mu + v * randn(gr.Nod,1)),gr.size);

mustSmooth = false; % true;
if mustSmooth
    SPAN = gr.Nx;
    Tx = Tx';
    Tx(:) = smooth(Tx(:),SPAN);
    Tx = Tx';
end
Tx(:,[1 2 end]) = mT;


%% Recharge
FQ    = gr.const(0); FQ(:,1) = 2 * gr.dy * gr.dz .* Peff(:,1);

IH  = gr.const(0); IH(:,1)=0.5*diff(gr.xGr([1 end]));

%% IBOUND (specifies where heads are fixed)
IBOUND = ones(gr.size);
IBOUND(:,end) = FXHD;

%% Run the flow model
[Phi,Q,Qx,Qy,Psi] = fdm2(gr,Tx,Tx,IBOUND,IH,FQ);

%% Setup the particle tracking model
t      = 0:50:12500; % times

aL     =  0; % [ m ] longitudinal dispersivity
aT     =  0.1; % [ m ] transversal dispersivity
Diff   =  1e-4; % [m2/d] diffusion coefficient
R      =  1; % [ - ] retardation
lambda =  0; % [1/d] decay

%% Generate starting particles and run MOC
swarm      = true;   % a swarm of particles

if swarm
    Np =5000;
    x =  gr.xGr(2) + diff(gr.xGr([2   3]))*rand(Np,1);
    y =  gr.yGr(1) + diff(gr.yGr([1 end]))*rand(Np,1);

    P = Moc(gr,Qx,Qy,Peff,R,t,x,y,'Diff',Diff,'aL',aL,'aT',aT,'lambda',lambda);

else  % uniformly distributed particles (nxn in each cells)
    n = 3;

    P = Moc(gr,Qx,Qy,Peff,R,t,n,'aL',aL,'aT',aT,'Diff',Diff,'lambda',lambda);

end

%% Visualize results
close all;

figure('pos',screenPos(1,0.3));
ax = axes('nextplot','add','xlim',gr.xGr([1 end]),'ylim',gr.yGr([end 1]));

xlabel(ax,'x [m]'); ylabel(ax,'y [m]');
title(ax,'example2: MOC, flow from left to right');

% Contour log conductivities

kRange = min(Tx(:)):(max(Tx(:))-min(Tx(:)))/50:max(Tx(:));
set(ax,'clim',log(kRange([1 end])));

surf(ax,gr.xGr,gr.yGr,-ones(gr.Ny+1,gr.Nx+1),augment(log(Tx)),'edgeColor','none');

% add colobar
hb = colorbar(); set(get(hb,'title'),'string','ln(k)');

% add stream lines
prange = min(Psi(:)):(max(Psi(:))-min(Psi(:)))/50:max(Psi(:));
contour(ax,gr.xp,gr.yp,Psi,prange,'color','w');

% Show moving particles (stored in struct P)
time = [P.time];
Np   = numel(P(1).x);
for it=1:numel(time)
    % Title will change according to passing time
    ttl = sprintf('Tracking %d particles time = %.0f d',Np,time(it));
    if it==1
        % First loop, title and plot
        ht = title(ax,ttl);
        h = plot3(ax,P(it).x,P(it).y,50*ones(size(P(it).x)),'k.');
    else
        % Subsequent loops, reset title and points
        set(ht,'string',ttl);
        set(h,'xData',P(it).x,'yData',P(it).y);
        drawnow();  % necessary to update plot
        pause(0.1); % smooth movie
    end
end
    
%% Plot arrival times of particles at different distancees

PX = [P.x];
xx = gr.xm(1) + diff(gr.xm([1 end]))* (0.2:0.2:0.8);
N = zeros(numel(time),numel(xx));
for it=1:numel(time)    
    for ix=1:numel(xx)
        N(:,ix) = sum(PX>xx(ix),1);
    end
end

clr = 'brgkmcy';
figure; hold on; grid on;
ht = title(sprintf('Nr of particles in cells, total simulated = %d',Np));
xlabel('time'); ylabel('total nr of particles');
for ix = 1:numel(xx)
    plot(time,N(:,ix),clr(ix));
end

%% Check water balance
fprintf('Water balances:\n');
fprintf('Total water balance = %10g (should be zero)\n',sum(Q(IBOUND~=0)));
fprintf('Total recharge (active  cells) = %10.0f m3/d\n',sum(Q(IBOUND>0)));
fprintf('Total discharge(fixhd + wells) = %10.0f m3/d\n',sum(Q(IBOUND<0)));

