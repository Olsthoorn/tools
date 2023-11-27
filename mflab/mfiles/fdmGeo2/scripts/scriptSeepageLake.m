%%% modelScript -- sets up 2D FDM, runs it and shows results

FIXED = -1;   % indicatre fixed head node in IBOUND

%% Grid
d   = 25; % cell width, also try 5 m
xGr = -75:1:0;
yGr = -20:1:7;
gr  =  grid2DObj(xGr,yGr);

%% Island
BASIN  =  gr.Xm<-60 & gr.Ym>6;
POND   =  gr.Xm>-15 & gr.Xm<-0 & gr.Ym>5;
DRAIN  =  gr.Xm>-1  & gr.Ym<1   & gr.Ym>-1;

IBOUND = ones(gr.size);
IBOUND(BASIN) = FIXED;

%% Transmissivities
kx = 10; ky = kx;

Kx = gr.const(kx);
Kx(POND) = 10000;
Peff    = gr.const(0.35); Peff(POND) = 1;
%% Start heads, initial heads
hBasin = 7;
hL     = 3;
IH     = gr.const(hL); IH(BASIN)=hBasin;

%% Fixed inflows of cells
rch       = 0.01; % net recharge rate
FQ        = gr.const(0);
FQ(1,:)   = gr.dx * rch;
FQ(DRAIN) = -5/sum(gr.Area(DRAIN)); 

%% Run model
[Phi,Q,Qx,Qy,Psi] = fdm2(gr,Kx,Kx,IBOUND,IH,FQ);

aL= 0.01; aT = aL/10; Diff=1e-4;
N = 10000;
t  = 1:1:200;
R  = 1;
xp = min(gr.Xm(BASIN)) +rand(N,1)*(max(gr.Xm(BASIN))-min(gr.Xm(BASIN)));
yp = gr.yGr(2) +rand(N,1)*0;

P = Moc(gr,Qx,Qy,Peff,R,t,xp,yp,'aL',aL,'aT',aT,'Diff',Diff);

%% Visualize results
figure('pos',screenPos(1,0.2)); hold on;
xlabel('x [m]'); ylabel('y [m]');
title('Cross with seepage lake');

hRange = floor(min(Phi(:))):0.1:ceil(max(Phi(:)));

set(gca,'clim',hRange([1 end]));

contourf(gr.xm,gr.ym,Phi,hRange,'edgeColor','none');

contour(gr.xp,gr.yp,Psi,20,'color',[0.8 0.8 0.8]);

hb=colorbar; set(get(hb,'title'),'string','head [m]');

for it=1:numel(P)
    ttl = sprintf('time = %.0f',t(it));
    if it==1
        ht = title(ttl);
        hp = plot(P(it).x,P(it).y,'k.');
    else
        set(ht,'string',ttl);
        set(hp,'xData',P(it).x,'yData',P(it).y);
        drawnow;
        %pause(0.2);
    end
end
    