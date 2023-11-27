%% Example comparing the the axially symmetric model with De Glee solution
close all

Qo    = -2400;
kD    =  500;
c     =  350;
lambda= sqrt(kD*c);

xGr = logspace(0,4,41);
yGr = [0,-1,-2];
gr  = grid2DObj(xGr,yGr,'AXIAL',true);

k       = gr.const([ 0.5*gr.dy(1)/c;   kD/gr.dy(2) ]);

IBOUND = gr.const(1); IBOUND(1,:)= -1;
IH     = gr.const(0);
FQ     = gr.const(0); FQ(end,1)=Qo;

[Phi,Q,Qx,Qy,Psi]=fdm2a(gr,k,k,IBOUND,IH,FQ);

fi1 = Qo/(2*pi*kD) * besselk(0,gr.xm/lambda);
fi2 = Qo/(2*pi*kD) * log(1.123*lambda./gr.xm);

figure; hold on; title('Comaring model with De Glee');
xlabel('r [m]'); ylabel('s [m]');
set(gca,'xScale','log');

plot(gr.xm,Phi(end,:),'ro');
plot(gr.xm,fi1       ,'b');
plot(gr.xm,fi2       ,'k');

legend('fdm','De Glee','approx De Glee');
