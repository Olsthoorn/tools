%% Theis drawdown with arbirary extraction, by convolution of analytical solution

%% Aquifer parameters and distance of piezometer
kD = 900;  % m2/d
S  = 0.2;  % [-] storage coeff (specific yield)
r  = 150;  % distance of piezometer from well

%% Show different analytical responses for Theis
tau = logspace(-1,3,41);  % log scale for response

% SR IR and BR as anonymous functions
SR   = @(tau) 1/(4*pi*kD) * expint(r^2*S./(4*kD*tau));    % step response
IR   = @(tau) 1/(4*pi*kD) * exp(-r^2*S./(4*kD*tau))./tau; % impulse response
BR   = @(tau,dTau) SR(tau)-SR(max(tau-dTau,tau(1)));      % block response

% Check by computing SR from IR and the other way around
SR2  = @(tau) cumsum([SR(mean(tau([1 2]))) 0.5*(IR(tau(1:end-1))+IR(tau(2:end))).*diff(tau)]);
IR2  = @(tau) (SR(tau(2:end))-SR(tau(1:end-1)))./diff(tau);

taum = 0.5*(tau(1:end-1)+tau(2:end)); % halfway between times

%% Visualize

% Defaults for creating plot axes
defaults={'nextPlot','add','xGrid','on','yGrid','on','xScale','log'};

%% Show tesp response and check
figure; axes(defaults{:});
xlabel('tau [d]'); ylabel('step response [d/m2]'); title('step response');
plot(tau,SR( tau),'b','lineWidth',2);  % SR
plot(tau,SR2(tau),'ko');               % SR check
legend('SR','SRcheck');

%% Show impulse response and check
figure; axes(defaults{:});
xlabel('tau [d]'); ylabel('impulse response [1/m2]'); title('Impulse response');
plot(tau ,IR( tau),'b');  % IR
plot(taum,IR2(tau),'mo'); % IR check
legend('IR','IRcheck');

%% Block response
dTau = [0.1 1 10 100]; % Width of block response

figure; axes(defaults{:});
set(gca,'xScale','log'); xlabel('tau [d]'); ylabel('SR m/(m3/d)');
title('step response and block response for different Dtau');
plot(tau,SR(tau),'b');  % Step response

leg={'SR'};
for ib=1:numel(dTau)  % block responses
      plot(tau,BR(tau,dTau(ib)),mf_color(ib,'rgkmc'));
      leg{end+1}=sprintf('SR(Dtau=%.1f',dTau(ib)); %#ok
end
legend(leg{:});

%% Linear scale of block response for convolution
dTau     = 1;  % time steps of 1 day
tauLin   = 0:dTau:tau(end);
tauLin(1)=0.0001; % first value > 0 to prevent NaN;
br       = BR(tauLin,dTau); % compute block response

%% For different inputs for convolution
Q0= 1200;
Q1 = Q0*ones(size(br));                              % constant
Q2 = Q0*sort(ceil(5*rand(size(br))))/5;              % 5 steps
Q3 = Q0*5*rand(size(br))/5;                          % pure random
Q4 = Q0*sin(2*pi*3*(1:numel(tauLin))/numel(tauLin)); % sine input

%% Visualize
figure; axes(defaults{:}); set(gca,'xScale','lin','yDir','reverse');
xlabel('time [d]');
ylabel('s [m]');
title('Theis drawdown with arbitrary extraction, by convolution');

plot(tauLin,filter(br,1,Q1),'r');
plot(tauLin,filter(br,1,Q2),'b');
plot(tauLin,filter(br,1,Q3),'k');
plot(tauLin,filter(br,1,Q4),'g');
legend('Q constant','Q 5 equal steps','Q pure random','Q sine');