%%% ModelScript -- Example 4 sets up 2D FDM, runs it and then runs 
%   Runs a particle tracking model (Moc method of characteristics)
%   Shows the results
%   TO 140417

%% Cleanup
close all;

%% Constants used in IBOUND
FXHD     = -1;
INACTIVE =  0;
ACTIVE   =  1;

clr = 'brgkmcy';  % list of colors

%% Generate the Grid for th FDM flow modoel
xGr   = 0:500:5000;
yGr   = 0:500:500;   % 5000;
zGr   = [0 -100];
gr    =  grid2DObj(xGr,yGr,zGr); % generates 2D gridObj with depth

% Show the grid
% figure; hold on;
% xlabel('x [m]'); ylabel('y [m]');
% title('example2: MOC, flow from left to right');
% gr.plot('c');  % lines in cyan color

%% Transmissivities
Tx = gr.const(600);
Ty = gr.const(600);

%% Recharge
rch   = 0.0;   % 01; % net recharge rate
FQ    = gr.Area * rch;

%% Wells
well = [1500,2000,-2400
        3500,3500,-2400];
    

%Idwell     = gr.Idx(well(:,1),well(:,2));
%FQ(Idwell) = well(:,3);

IH  = gr.const(0); IH(:,1)=20;

%% IBOUND (specifies where heads are fixed)
IBOUND = ones(gr.size);
IBOUND(:,[1 end]) = FXHD;
%IBOUND([1 end],:) = FXHD;

%% Run the flow model
[Phi,Q,Qx,Qy] = fdm2(gr,Tx,Ty,IBOUND,IH,FQ);

%% Setup the particle tracking model
t    = 0:500:150000; % times
%t    = 0:1:100;
Peff = gr.const(0.35); % effective porosity

aL   = 1;   % [ m ] longitudinal dispersivity
aT   = aL/10; % [ m ] transversal dispersivity
Diff = 1e-4;  % [m2/d] diffusion coefficient
R    = 2;     % [ - ] retardation
lambda = 2e-5;% [1/d] decay

%% Generate starting particles and run MOC
swarm      = true;   % a swarm of particles
pointSwarm = false;   % several point swamrs of particles

if swarm
    Np =250000;
    if ~pointSwarm
        x = (rand(Np,1)-0.5)*1000 + gr.xm(ceil(gr.Nx/2)) ;
        y = (rand(Np,1)-0.5)*1000 + gr.ym(ceil(gr.Ny/3)) ;
        
        %x(:) = mean(x); % debug
        
        P = Moc(gr,Qx,Qy,Peff,R,t,x,y,'Diff',Diff,'aL',aL,'aT',aT,'lambda',lambda);

    else % if pointSwarm
        xc = [1895   2160   2252   2656   3255   2586   1238];
        yc = [3019   3151   2770   2741   2595   1791   2770];
        for i=numel(xc):-1:1
            x((i-1)*Np+1:i*Np) = xc(i); 
            y((i-1)*Np+1:i*Np) = yc(i);
        end
        
        P = Moc(gr,Qx,Qy,Peff,R,t,x,y,'aL',aL,'aT',aT,'Diff',Diff,'lambda',lambda);

    end
else  % uniformly distributed particles (nxn in each cells)
    n = 3;
%    [P,Icells] = Moc(gr,Qx,Qy,Peff,R,t,n); % no dispersion, diffusion, decay

    P = Moc(gr,Qx,Qy,Peff,R,t,n,'aL',aL,'aT',aT,'Diff',Diff,'lambda',lambda);

end

%% Visualize results
close all;

figure('DoubleBuffer','on');
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
Np   = numel(P,x);
for it=1:numel(time)/2
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
    
break;

%% Plot cumulative number of particles captured by wells
figure; hold on; grid on;
ht = title(sprintf('Cumulative particles captured by the wells, total simulated = %d',Np));
xlabel('time'); ylabel('total nr of particles captured');

% Make array of cell nrs in which particles are after each time step
PIcell = [P.Icells];

leg = [];
if exist('Idwell','var')
    for iw = numel(Idwell):-1:1
        h(iw) = plot(t,sum(PIcell==Idwell(iw),1),clr(iw));
        leg{iw} = sprintf('well %d',iw);
    end
    legend(h,leg{:},2);
end
%% Plot total mass caputured by well where 1 particle represents 1 mass unit

% Update title of previous graph using handle ht.
set(ht,'string',sprintf('%s, with and without decay',get(ht,'string')));

% This mass is subject to decay
if isfield(P,'mass')
    clr = 'brgkmcy';

    mass   = [P.mass];
    for iw = numel(Idwell):-1:1
        % Add to previous plot for comparison
        h(iw) = plot(t,sum(mass.*(PIcell==Idwell(iw)),1),clr(iw));
    end
    % Also refer to legend of previous plot
end

%% Plot vectors indicating flow direction and strength

if gr.Ny>1 && gr.Nx>1
    figure; hold on;
    xlabel('x [m]'); ylabel('y [m]');
    title('Flow model with Quiver');

    % Show arrows of flow direction and magnitude
    qx = [Qx(:,1), Qx, Qx(:,end)]; qx = 0.5*(qx(:,1:end-1) + qx(:,2:end));
    qy = [Qy(1,:); Qy; Qy(end,:)]; qy = 0.5*(qy(1:end-1,:) + qy(2:end,:));

    % Contour heads
    phiMax = max(Phi(:)); phiMin = min(Phi(:));  hRange = phiMin:(phiMax-phiMin)/25:phiMax;

    contourf(gr.xm,gr.ym,Phi,hRange,'edgeColor','none');
    quiver(gr.Xm,gr.Ym,qx,qy);

    hb = colorbar; set(get(hb,'title'),'string','head [m]')  % Colorbar
end

%% Check water balance
fprintf('Water balances:\n');
fprintf('Total water balance = %10g (should be zero)\n',sum(Q(IBOUND~=0)));
fprintf('Total recharge (active  cells) = %10.0f m3/d\n',sum(Q(IBOUND>0)));
fprintf('Total discharge(fixhd + wells) = %10.0f m3/d\n',sum(Q(IBOUND<0)));

