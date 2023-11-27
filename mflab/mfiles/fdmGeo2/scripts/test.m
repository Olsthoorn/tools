%% ARIMA model on the results
a = -0.7;                    % damping at lag 1
Y1 = filter(b,[1 a],X1) + d; % generate noiseless output
Y2 = filter(b,[1 a],X2) + d; % noisy putput

c = [0.1 0.001 0.001];       % initial c-vector

F = @(p)  ARIMA(p,Y2-Y1);    % define function F for lsqnonlin

[c,resNorm,FVAL,EXITFLAG] = lsqnonlin(F,c,zeros(size(c)));

[e,yHat] = ARIMA(c,Y1-Y2);   % compute e using final c-vector

figure; hold on; title('error from Arima'); xlabel(yrs); ylabel('e [m]');
plot(t,e);
std(e);
