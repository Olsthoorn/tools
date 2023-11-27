% Large-diameter well drawdown with well storage (Bouton, 1963)

%% Specify well
rW  = 5;
zW  =-5;
k   = 1;
ss  = 0.000;
Sy  = 0.05;
Qw  =-130;

%% Grid
rGr = [logspace(0,4,50), rW+[-0.5 -0.25 -0.1 0 0.1 0.25 0.5 1]];
yGr = [0:-1:-20, zW+[-0.5 -0.25 -0.1 0 0.1 0.25 0.5]];
t   = logspace(0,3,31);

gr = grid2DObj(rGr,yGr,'AXIAL',true);

%% Define inWell domain
inWell = gr.Ym>zW & gr.Rm<rW;

%% Arrays
IBOUND = gr.const(1);
K  = gr.const(k); K(inWell)=10000;
Ss            = gr.const(ss);  % elastic storage 
Ss(1,:)       = Sy/gr.dy(1);   % specific yield
Ss(1,gr.rm<rW)= 1/gr.dy(1);    % within well Ss=1 (free surface)

IH = gr.const(0);
FQ = gr.const(0); FQ(1,2)=Qw;

%% Run model
[Phi,Qt,Qr,Qz,Qs]=fdm2t(gr,t,K,K,Ss,IBOUND,IH,FQ);

%% Analytic solution Theis
kD = sum(K( :,end) .* gr.dy);
SY = sum(Ss(:,end) .* gr.dy);
fi = Qw/(4*pi*kD)*expint(bsxfun(@rdivide,gr.rm.^2*SY,4*kD*t(:)));

%% Visualize
close all
figure; axes('nextPlot','add','xScale','log');
title('Drawdown insize pp large diameter well, compare with Theis');
xlabel('r [m]'); ylabel('dd [m]');
plot(gr.rm(:),fi,'k');
plot(gr.rm,squeeze(Phi(1,:,:)),'+');
