%% modelScript -- transient model for Theis well case
% Transient axially symmetric model simulating transient extraction
% from a confined aquifer according to Theis. Comparison with analytic
% solution.

close all;

%% Data for theis well and aquifer
Qw =-2400;
rW = 0.1;
kD = 200; D=20; k = kD/D;
S  = 0.1;       Ss = S/D;

t=logspace(-3,2,51);

%% grid
rGr = logspace(log10(rW),2,41);
yGr = [0  -20];
gr = grid2DObj(rGr,yGr,'AXIAL');

IBOUND = gr.const(1);
IBOUND(:,end)=-1;

HI = gr.const(0);
FQ = gr.const(0);  FQ(1,1) = Qw;
    
TheisR = @(t) Qw/(4*pi*kD) * expint(gr.rm.^2*S./(4*kD*t));
TheisT = @(r) Qw/(4*pi*kD) * expint(   r .^2*S./(4*kD*t));

[Phi,B]=fdm2t(gr,t,k,k,Ss,IBOUND,HI,FQ);

%% Check storage
for idt = numel(diff(t)):-1:1
    QIN(idt,1) = sum(B(idt).NET(:));
    QFH(idt,1) = sum(B(idt).FH(:));
    QFQ(idt,1) = sum(B(idt).FQ(:));
    QST(idt,1) = sum(B(idt).STO(:));
end

format shortg
fprintf('%12s %12s %12s %12s %12s\n','Qin','QFH','-FQ','QST','QFH+FQ-QST');
display([QIN -QFH -QFQ QST QFH+QFQ-QST]);

%% Drawdown as function of distance
close all
figure;

subplot(2,1,1,'nextplot','add','xScale','log');
title( 'Theis drawdown as function of time for different r');
xlabel('t [d]'); ylabel('dd [m]');

for ir=1:5:gr.Nr
   plot(t,squeeze(Phi(1,ir,:)),'bo'); % numerical 
   plot(t,TheisT(gr.rm(ir)),'k'); % analytical
end

subplot(2,1,2,'nextplot','add','xScale','log');
title( 'Theis drawdown as function of r for different times');
xlabel('r [m]'); ylabel('dd [m]');

for it=5:5:length(t)
   plot(gr.rm,Phi (1,:,it)  ,'bo'); % numerical
   plot(gr.rm,TheisR(t(it)), 'k'); % analytical
end
