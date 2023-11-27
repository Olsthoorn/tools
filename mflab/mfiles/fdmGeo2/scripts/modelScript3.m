%%% ModelScript -- Example 3 sets up 2D FDM with massive particle tracking
% uses fdm2 and Moc

%% Consants used with IBOUND
FXHD     = -1;
INACTIVE =  0;
ACTIVE   =  1;

%% Grid
xGr   = 0:100:5000;
yGr   = 0:100:5000;
zGr   = [0 -100];
gr    =  grid2DObj(xGr,yGr,zGr);

%% Transmissivities
Tx = gr.const(6);
Ty = gr.const(6);

%% Recharge
rch   = 0.005; % net recharge rate
FQ    = gr.Area * rch;

%% IBOUND
IBOUND = gr.const(ACTIVE);
IBOUND (13,17) = FXHD;    % well

IH = gr.const(0);

close all;

%% Run model
[Phi,Q,Qx,Qy] = fdm2(gr,Tx,Ty,IBOUND,IH,FQ);

t    = 0:50:5000;
Peff = gr.const(0.35);
R    = 1;
n    = 1;
aL = 10;
P = Moc(gr,Qx,Qy,Peff,R,t,n);
%P = Moc(gr,Qx,Qy,Peff,R,t,n,'aL',aL);
%% visualize results
defaults = {'nextplot','add','xlim',gr.xGr([1 end]),'ylim',gr.yGr([end 1])};

% set up figure
figure; axes(defaults{:}); %#ok
xlabel('x [m]'); ylabel('y [m]');
title('example2: MOC, flow from left to right');

%set(gca,'xlim',[1000 2000],'ylim',[3000 4000]);
% contour heads
phiMax = max(Phi(:)); phiMin = min(Phi(:));
hRange = phiMin:(phiMax-phiMin)/25:phiMax;
contourf(gr.xm,gr.ym,Phi,hRange,'edgeColor','none');

% show particles
time = [P.time];
for it=1:numel(time)
    ttl = sprintf('particles time = %.0f d',time(it));
    if it==1
        ht = title(ttl);
        h = plot(P(it).x,P(it).y,'b.','markersize',5);
    else
        set(ht,'string',ttl);
        set(h,'xData',P(it).x,'yData',P(it).y);
        drawnow();
        pause(0.2);
    end
end
      
%% Visualize results

figure; hold on;
xlabel('x [m]'); ylabel('y [m]');
title('Head with  Quiver');

% contour heads
phiMax = max(Phi(:)); phiMin = min(Phi(:));
hRange = phiMin:(phiMax-phiMin)/25:phiMax;
contourf(gr.xm,gr.ym,Phi,hRange,'edgeColor','none');


%Show arrows of flow direction and magnitude
qx = [Qx(:,1), Qx, Qx(:,end)]; qx = 0.5*(qx(:,1:end-1) + qx(:,2:end));
qy = [Qy(1,:); Qy; Qy(end,:)]; qy = 0.5*(qy(1:end-1,:) + qy(2:end,:));
quiver(gr.Xm,gr.Ym,qx,qy);

hb = colorbar; set(get(hb,'title'),'string','head [m]')  % Colorbar

%% Check water balance
fprintf('Water balances:\n');
fprintf('Total water balance = %10g (should be zero)\n',sum(Q(IBOUND~=0)));
fprintf('Total recharge (active  cells) = %10.0f m3/d\n',sum(Q(IBOUND>0)));
fprintf('Total discharge(fixhd + wells) = %10.0f m3/d\n',sum(Q(IBOUND<0)));

