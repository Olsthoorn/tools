x = -0.5:0.01:0.5;
sigma = 0.1;

figure;
axes('nextPlot','add','xgrid','on','ygrid','on','fontsize',12);
ylabel('h [m]');
xlabel('erfc, ierfc');

plot(0.5*erfc(-x/sigma),x,'b','lineWidth',1);
plot(0.5*ierfc(-x/sigma,1),x,'r','linewidth',2);

title('erfc en ierfc functies, voor sigma=0.1 [m]');
legend('0.5 erfc(-h/sigma)','0.5 ierfc(-h/sigma)');