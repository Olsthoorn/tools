function wh = Wh(u,rho)
%WH -- Hantush's well function (semi-confined transient flow to a well'
% The Hantush function is computed by numerical integration
%
% USAGE: wh = Wh(u,rho)
%
%  u   = array with values of r^2*S/*4*kD*t;
%  rho = array r/lambda, same size as u
%
% TO 140428

if ~all(size(u)==size(rho))
    error('size(u) ~= size(rho)');
end

wArg = @(xi) exp(-exp(xi)-exp(-xi).*((rho(:).^2/4) * ones(1,size(xi,2))));

sz = size(u);   % szie of u, to be reshaped at the end
UB = 20;        % upper boundary of integration
LB = log(u(:)); % lower boundary of integration

dksi = 0.01; N = (UB - min(LB))/dksi; % intgration step size

M  = numel(LB);
N  = round(N);

KSI = cumsum([ LB dksi* ones(M,N) ],2); % argument values

wh      = wArg(KSI) * dksi;  % argument times dksi
wh(:,1) = 0.5*wh(:,1);       % trapezium rule, first step
wh      = sum(wh,2);         % trapezium rule
wh      = reshape(wh,sz);    % reshape to original format