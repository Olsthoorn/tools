%% Cleanup

clear
close all

shapesDir = '../duneData/';
%% Default properties

% for labels
lblDflt = {2,2, 'color','k','rotation',30,'interpreter','none'};

% axes extent on new figure
axSz    = [0.8 0.8];  % default axis size

%% Hyrology
rch = 0.000; % [m/d] recharge
kh  = 7.5;   % [m/d] horizontal conductivity
base= -15;   % base elevation of first aquifer

%% Preparing the model

% The shape objects to be used in this model
ponds  = shapeObj([shapesDir 'GeulenRondDr9-10']   ,'-v');  % -v means non-verbose = be silent
canals = shapeObj([shapesDir 'KanalenRondDr9-10']  ,'-v');
drains = shapeObj([shapesDir 'Dr9-10']             ,'-v');
bores  = shapeObj([shapesDir 'BoreHolesNearDr9-10'],'-v');

%% Headers and tables used to set new attributes or replae values

% specify change table for ponds
pondsHdr = {'GEULEN_AWD' 'CODE','h','c' 'hBot'};
pondsTable = {
	16  18  5.21 0.71  4.4
	17  16  5.69 0.71  4.9
	18  14  6.25 0.71  5.5
    19  12  6.71 0.71  5.9
    21  10  6.76 0.71  5.9
	27  7   5.49 0.71  4.7
	30  8   6.82 0.71  6.0
	31  6   6.04 0.71  5.2
    32  43  6.76 0.71  5.9
    33  3   6.72 0.71  5.9
	35  4   6.72 0.71  5.9
	36  5   6.58 0.71  5.8
    37  20  5.32 0.71  4.5
    39  45  6.93 0.71  5.1
    };

% change table for canals
canalsHdr   = { 'CODE' 'NAME' 'h' 'c' 'KANAAL_AWD'};
canalsTable = {
    5 'Westerkanaal'   0.80   0.71  16
    5 'Westerkanaal'   1.00   0.71  28
    2 'Barnaert'       3.55   0.71  23
    8 'Barnaert'       3.55   0.71  25
    30 '1000m'         3.55   0.71  27};


% change table for drains
drainsHdr = {'UBAK'	'LENGTH' 'NAME' 'DRAINS_AWD' 'h'  'hD' 'c'};
drainsTable = {
	4	93.07  'C030'     98  2.57 3.10 2 
	4	190.7  'C029a'    99  2.50 NaN  2 % ??
	4	50.99  'C029'    100  2.97 4.48 2
 	5	47.26  'C028'    101  3.46 NaN  2
	5	192.81 'C028a'   102  3.00 NaN  2 % ??
	5	151.49 'C027a'   103  3.00 NaN  2 % ??
	5	53.59  'C027'    104  3.35 NaN  2 };

%% Add new attributes and change atrribute values based on tables above using CODE as Key

ponds  = ponds. setAttr('GEULEN_AWD',pondsHdr ,pondsTable,'select'); 
canals = canals.setAttr('KANAAL_AWD',canalsHdr,canalsTable,'select');
drains = drains.setAttr('DRAINS_AWD',drainsHdr,drainsTable,'select');

%% Get DEM and fill in the NaNs (open water space)

% Get DEM as from GIS
[DEM,xGr,yGr]=readASC([shapesDir 'DEMDr9-Dr10.asc']);
xm     = 0.5*(xGr(1:end-1)+xGr(2:end));
ym     = 0.5*(yGr(1:end-1)+yGr(2:end));
[Xm,Ym]= meshgrid(xm,ym);
ok     = ~isnan(DEM);
F      = TriScatteredInterp(Xm(ok),Ym(ok),DEM(ok));
DEM    = F(Xm,Ym);


%% Geneate the final grid
gr = grid2DObj(xGr,yGr,   cat(3,DEM,base*ones(size(DEM)))  ,'LAYCON',true );

%% Visualize the shapes

% Defaults for next axes
axDflt  = {'nextPlot','add','xLim',gr.xGr([1 end]),'ylim',gr.yGr([end 1])};


figure('pos',screenPos(axSz)); axes(axDflt{:}); %#ok

% Plot shapes
ponds. plot('m');
canals.plot('b');
drains.plot('k')
bores. plot('ro','markerfacecolor','r');

% label them
ponds. label('code'   ,lblDflt{:});
canals.label('code'   ,lblDflt{:});
drains.label('length' ,lblDflt{:})
bores. label('BOORGAT',lblDflt{:});

% Add grid cell numbers to objects
drains = drains.Idx(gr);
ponds  = ponds. Idx(gr);
canals = canals.Idx(gr);
bores  = bores. Idx(gr);

% Get stress arrays
DRN = drains.stress('DRN',gr);
GHB = canals.stress('GHB',gr);
RIV = ponds. stress('RIV',gr);

% Use object 'stage' attribute as initial head
IH     = gr.    const(0);
IH     = ponds. getA(IH,'h');
IH     = canals.getA(IH,'h');
IH     = drains.getA(IH,'h');
% Set IBOUND witin shape to fixed head
IBOUND = gr.    const(1);


% Fixed flows
FQ     = gr.Area *rch;

% Conductiviy
K      = gr.const(kh);

%% Run the model
fixHds = true;

if fixHds
    % FIXED HEADS -- set IBOUND for the objects with fixed heads
    IBOUND = ponds. getA(IBOUND,-1);
    IBOUND = canals.getA(IBOUND,-1);
    IBOUND = drains.getA(IBOUND,-1);

    [Phi,Q,Qx,Qy] = fdm2a(gr,K,K,IBOUND,IH,FQ,'LAY',true);
else    
    [Phi,Q,Qx,Qy] = fdm2a(gr,K,K,IBOUND,IH,FQ,'dr',DRN,'ghb',GHB,'riv',RIV);
end
%% Visualize heads

figure('pos',screenPos(axSz));  axes(axDflt{:});

%colormap('pink');

set(gca,'xlim',xGr([1 end]),'ylim',yGr([end 1]),'clim',[-10,10]);

xlabel('x [m RD]'); ylabel('y [m RD]');

ttl = 'Head near drains 9 and 10 in AWS dunes';

title(ttl);

hRange = floor(min(Phi(:))):0.5:ceil(max(Phi(:)));

% Contour
contourf(gr.xm,gr.ym,Phi,hRange);

% Show objects on top
ponds. fill('c','lineWidth',0.1,'faceAlpha',0.5);
canals.fill('b','lineWidth',0.1,'faceAlpha',0.5);
drains.plot('k','lineWidth',0.1)
bores. plot('ro','markerfacecolor','r');

% Add colorbar with correct title
hb = colorbar; set(get(hb,'title'),'string','head [NAP]');

% settings transport model

t    = 0:365; % times
Peff = gr.const(0.35); % effective porosity

aL   = 10;   % [ m ] longitudinal dispersivity
aT   = aL/10; % [ m ] transversal dispersivity
Diff = 1e-4;  % [m2/d] diffusion coefficient
R    = 1;     % [ - ] retardation
lambda = 2e-5;% [1/d] decay

% n     = 5;
% [x,y] = ponds.points(n,gr);

dL    = 1;
[x,y] = ponds.edgePoints(dL);

P = Moc(gr,Qx,Qy,Peff,R,t,x,y,'aL',aL,'aT',aT,'Diff',Diff); %,'lambda',lambda);

%% movie showing the particle move from the ponds to the drains

for it=1:numel(P)
    tttl = [ttl sprintf(', time = %.0f d',t(it))];
    if it==1 % first cycle, set up the plot
        ht = title(tttl);
        hp = plot(P(it).x,P(it).y,'k.');
    else % subsequent cycles only update their positions
        set(ht,'string',tttl); % update title (with new time)
        set(hp,'xData',P(it).x,'yData',P(it).y);
        drawnow();    % required of set is used instead of plot
    end
end

%% Show the points captured by the drains
N = drains.captured(P);
figure; hold on; xlabel('time [d]'); ylabel('particles captured by drains');
plot(t,N),'b';

%% Show the points capured by the canals
N = canals.captured(P);
plot(t,N,'r');

%% Show the points captured by drain 1.
N = drains(1).captured(P);
plot(t,N,'g');

%% Visualizaation of head relative to DEM
if 0
    figure('pos',screenPos(axSz)); 

    ax1=axes(axDflt{:},'clim',[-1 8]);
    ax2=copyobj(ax1,gcf); set(ax2,'color','none','clim',[-1 8]);

    xlabel('x [m RD]'); ylabel('y [m RD]');
    title({'CIE5440, Geohydrology 2, May 2014';...
        'Groundwater depth near drains 9 and 10 in the Amsterdam Dune Area'});

    Depth = DEM-Phi;
    hRange = floor(min(Depth(:))):0.5:ceil(max(Depth(:)));

    % Contour head above DEM
    contourf(ax1,gr.xm,gr.ym,Depth,hRange,'edgeColor','none');

    % [~,h] = contourf(ax2,gr.xm,gr.ym,Depth,[0 0],'EdgeColor','w');
    % set(get(h,'children'),'faceColor','b');
    % set(ax2,'position',get(ax1,'position'));

    % Show shapes
    ponds. fill('b','lineWidth',0.1);
    canals.fill('b','lineWidth',0.1);
    drains.plot('k','lineWidth',0.1)
    bores. plot('ro','markerfacecolor','r');

    % Add colorbar with correct title
    hb = colorbar; set(get(hb,'title'),'string','drawdown [NAP]');
end

%% Water balance
Qin  = Q; Qin( Q<0)=0;
Qout = Q; Qout(Q>0)=0;
fprintf('ponds  in: %10.1f  out: %10.1f total: %10.1f m3/d\n',ponds.getQ(Q));
fprintf('canals in: %10.1f  out: %10.1f total: %10.1f m3/d\n',canals.getQ(Q));
fprintf('drains in: %10.1f  out: %10.1f total: %10.1f m3/d\n',drains.getQ(Q));
Qfh = Q(IBOUND<0);
fprintf('fxd    in: %10.1f  out: %10.1f total: %10.1f m3/d\n',sum(Qfh(Qfh>0)),sum(Qfh(Qfh<0)),sum(Qfh));
fprintf('fxQ    in: %10.1f  out: %10.1f total: %10.1f m3/d\n',sum(FQ( FQ >0)),sum(FQ( FQ <0)),sum(FQ(:)));


