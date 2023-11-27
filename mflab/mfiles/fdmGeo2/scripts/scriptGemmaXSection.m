%% Xsection generating flow in cross section between pond 6 and Westerkanaal
%  along the pline where Gemma Serre Prat did her research in 2012
%
% TO 140517

clear
close all

shapesDir = '../shapes/';
%% Hyrology

rch = 0; % [m/d] recharge

%% Preparing the model

%% If needed digitize points in the image of the cross section using
%xy = geoRef('../shapes/CrossSec10J536-537.png',[0 4; 92 -16]);

%% Digitized and measured data

zGr = [ % NAP materialNr (layer tops and material thereunder)
      6.0   1 % pond level
      5.2   1 % pond bottom
      0.8   1 % canal level
     -0.2   1 % canal bottom
     -4.0   2 % clay 1
     -7.0   3 % clay 2
     -8.2   1 % sand
    -14.2   2 % clay 1
    -16.0   1 % sand
    -18.2   4 % bottom
    ];

kh = [ % materialNr kh
    1 10    % dune sand
    2 0.1   % clay 1
    3 0.1   % clay 2
    4 1e-6  % base
    5 0.71  % pond and canal bottom
    ];  

xPnt = [  % x, distance from left-most piezometer
      -34.0  % left end of model
      -23.0  % left Wcanal
      -13.0  % right Wcanal  % real width = 10 m
       95.5  % left P6   % real width =23 m
      118.5  % right P6
      150.5  % end of model
        ];
    
hdrs = {'name filter x z wdepth topTube'};
obs = { % name screenNr x z waterDepth topOfTube
  'PK05'   1 -10.0    0.8  0.00   0.8
  '10J336' 5  0.0  -13.7  0.15   2.863
  '10J336' 4  0.0   -8.7  0.18   2.878
  '10J336' 3  0.0   -6.4  1.05   2.874
  '10J336' 2  0.0   -2.6  1.58   2.882
  '10J336' 1  0.0   -0.8  1.02   2.892 
  'P65'    1  7.0    0.7  2.30   4.155
  'P64'    1 29.2    1.3  1.32   4.470
  'P63'    1 58.9    2.0  0.93   5.259
  'P62'    1 87.7    3.9  2.39   7.508
  '10J537' 5 91.7   -8.3  2.73   7.650
  '10J537' 4 91.7   -3.0  2.49   7.681
  '10J537' 3 91.7   -0.1  2.48   7.701
  '10J537' 2 91.7    2.9  2.47   7.716
  '10J537' 1 91.7    4.9  2.44   7.726
  'PG06'   1 95.5   6.04  0.00   6.04
  };


%% Generate the grid

xGr = xPnt(1):0.5:xPnt(end);  % entire model

% Or with mid of pond as water-divide close boundary
%xGr = xPnt(1):0.5:xPnt(end-2) + diff(xPnt([end-2 end-1]))/2;

yGr = zGr(end,1):0.2:zGr(1,1); % layer elevations

% Generate the grid2DObj, using LAYCON2 true. LAYCON2 sets teh layers to
% convertible in a 2D grid (y-direction). It signals that the model
% considers the grid as vertial with y in fact z values and that we have a
% water-table aquifer, with the water table elvation not know a priori.
% Hence the model will iterated. After each iteration cycle it checks which
% cells have a head below their bottom (condiering the y-coordinate as elevation)
% and switches those cells off. If  head are above the obttom of the cells,
% it switches those cells on.
gr = grid2DObj(xGr,yGr,'LAYCON2');

%% Visualize the grid with image of cross section

% The extent of the figure extent was obtained by digitizing two points
% and then compute the real world coordinates of the extents of this
% figure.
XYE = [
      -46.2       10.38
      150.4      -19.7
       ];

% Defaults properties for axes
axDflt  = {'nextPlot','add','xLim',XYE(:,1),'ylim',XYE(:,2)};

% Size of figure on screen
axSz = [0.8 0.8];

% Get figure of desired size
figure('pos',screenPos(axSz));

% Get the image with the cross section
if ismac
    A = imread([ shapesDir '../duneData/CrossSec10J536-537.png']);
else
    A = imread([ shapesDir '..\duneData\CrossSec10J536-537.png']);
end

% Show the image using the known extent of the figure axes
image(XYE(:,1),XYE(:,2),A);

% Make sure the image is not wiped out on successive plot instructions
hold on;

% Set image upside-up
set(gca,'yDir','normal');

% plot the grid with cyan grid lines
%gr.plot('c');

%% Boundary conditions

% Get material (i.e. sand, clay etc) number for each layer using the zGr
% as specified above and the now known elevations of the center of the
% layers. The trick is to % interpolate the layer numbers and then round
% down to get their integer values.
Imat = floor(interp1(zGr(:,1),zGr(:,2),gr.ym));

% With the layer numbers known, use this as indices into the kh list above
% to generate a full grid of conductivity values.
% You may want to look up how bsxfun works, it's really convenient,
% especialy when working in 3D. In our 2D case, we don't need it per se as
% we can use a matrix multiplication kh*ones(1,gr.Nx) as alternative.
K = reshape(kh(bsxfun(@times,gr.const(1),Imat(:)),2),gr.size);

%% Fixed heads 

% Get logicla arrays telling which cells belong the pond and the canal.
Pnd8 = gr.Xm>xPnt(end-2) & gr.Xm<xPnt(end-1) & gr.Ym> 5.2 & gr.Ym<6.0;
Wkan = gr.Xm>xPnt(    2) & gr.Xm<xPnt(    3) & gr.Ym>-0.2 & gr.Ym<0.8;


IH       = gr.    const(6);  % default
IH(Pnd8) = 6.04;    % set initial head for pond
IH(Wkan) = 0.80;    % set initial head for canal

% Set IBOUND witin shape to fixed head
IBOUND = gr.const(1);
IBOUND(Pnd8) = -1;  % pond  is fixed head
IBOUND(Wkan) = -1;  % canal is fixed head

% Fixed flows
FQ     = gr.const(0); % no fixed flows, i.e. all zero.

%% Run the model

% Also retrieve the Psi (stream function)
[Phi,Q,Qx,Qy,Psi] = fdm2a(gr,K,K,IBOUND,IH,FQ);

%% Visualize heads and stream lines

xlabel('x [m]'); ylabel('y [m]');

% title string
ttl = 'Head between pond 7 and Wcanal';

% title
title(ttl);

% compute convenient contour levels to draw
hRange = floor(min(Phi(:))):0.5:ceil(max(Phi(:)));

% fix the color range for axis
set(gca,'clim',[floor(hRange(1)) ceil(hRange(end))])

% Contour full color on the image
[~,h] = contourf(gr.xm,gr.ym,Phi,hRange);

% Make contour patches transparent
set(get(h,'children'),'faceAlpha',0.2);

% Compute convenient contour levels for stream function
pRange = min(Psi(:)):0.5:max(Psi(:));

% plot the stream lines
contour(gr.xp,gr.yp,Psi,pRange,'color','w');


%% plot the observation wells

% first extract the values from the given cell-array above
head = [[obs{:,3}]' [obs{:,4}]' [obs{:,5}]' [obs{:,6}]'];

% subtract measured wate depth from tube top elevation to get head
head = [head(:,[1 2]) diff(head(:,end-1:end),1,2)];

% plot label in red at location of piezometer with measured head
for ih = 1:size(head,1)
    text(head(ih,1),head(ih,2),sprintf('%.2f',head(ih,3)),'color','r');
end

% get computed head by interpolating in grid using piezometer coordinates
head(:,end+1) = interp2(gr.Xm,gr.Ym,Phi,head(:,1),head(:,2));

% plot label in blue at below label in red, now showing the computed head.
for ih = 1:size(head,1)
    text(head(ih,1),head(ih,2)-0.65,sprintf('%.2f',head(ih,end)),'color','b');
end

%% MOC transport model (particle tracking)

% settings transport model

t    = 0:5:720; % times to check position of particles
Peff = gr.const(0.35); % effective porosity

aL   = 0.1;  % [ m ] longitudinal dispersivity
aT   = aL/10; % [ m ] transversal dispersivity
Diff = 1e-4;  % [m2/d] diffusion coefficient
R    = 1;     % [ - ] retardation
lambda = 2e-5;% [1/d] decay

% n     = 5;
% [x,y] = ponds.points(n,gr);

dL    = 0.001;                    % distance between initial particle locations
xy = [xPnt(end-2) 5.2; xPnt(end-1) 5.2];       % full width of pond
%xy = [xPnt(end-2) 5.2; gr.xm(end) 5.2]; % half width of pond

% make a shape for a lne object (pond bottom) and generate starting point
% coordinates for the particles using shp.edgePoints(dL);
shp = makeShp('POLYLINE',xy);
[x,y] = shp.edgePoints(dL);

% run  Moc to get the position of the released particles over time
P = Moc(gr,Qx,Qy,Peff,R,t,x,y,'aL',aL,'aT',aT,'Diff',Diff); %,'lambda',lambda);

%% make movie (show movement of particles)

for it=1:numel(P)
    % title (changes because time changes)
    tttl = [ttl sprintf(', time = %.0f d',t(it))];
    if it==1  % first cycle
        ht = title(tttl);                % plot title
        hp = plot(P(it).x,P(it).y,'k.'); % plot particles 
    else      % all other cycles
        set(ht,'string',tttl);           % update title
        set(hp,'xData',P(it).x,'yData',P(it).y); % update particle coordinates
        drawnow();    % need drawnow() after update.
        pause(0.1);   % reduce speed for viewing
    end
end

%% Show cumulative arrival time of the particles

figure; hold on; xlabel('time [d]'); ylabel('fraction of particles captured');

% First count them, by summing the not active (already captured particles)
% at each time. All this info was stored in P by Moc
for it=numel(P):-1:1, active(it)=sum(~P(it).active); end

Np = numel(P(1).active);
plot(t,active/Np); % show cumulative arriaval time

%% Water balance

QIn    = sum(Q(Q>0));
QOut   = sum(Q(Q<0));
QfhIn  = sum(Q(IBOUND<0 & Q>0));
QfhOut = sum(Q(IBOUND<0 & Q<0));
fprintf('Qnet    in: %10.1f,  out: %10.1f,  sum: %10.1f m3/d\n',QfhIn,QfhOut,QfhIn+QfhOut);
fprintf('fxHd    in: %10.1f,  out: %10.1f,  sum: %10.1f m3/d\n',QIn  ,QOut  ,QIn  +QOut  );
