function [e,br,be,yHat]= Fun(p,t,X,Y)
%Fun -- convolution for lsqnonlin using analytical block response
% theta = A/t * exp(-alpha^2/(beta^2*t)-beta^2*t)
% y = A + w
% A = conv(Fun,p)
% w = conv(exp,a)
dt= mean(diff(t));
a = exp(p(1));    % exponential decay parameter noise model
A = exp(p(2));    % gain
a2= exp(p(3));    % parameter in br
b2= exp(p(4));   % parmaeter in br
d = exp(p(5));
fprintf('a = %g, A=%g a2=%g b2=%g d=%g\n',a,A,a2,b2,d);

% IR
t  = t - t(1)+dt;
br = A./t .* exp(-a2./t-b2.*t)*dt;
be = exp(-a * (1:numel(t))*dt)';

yHat = filter(br,1,X) + d;
e    = Y - yHat;

for it=2:numel(t)
%      yHat(it) = sum(br(1:it).*X(it:-1:1))+ d; %a * e(it-1) + d;
%       e(it)    = Y(it) - yHat(it) - a * e(it-1);
%        e(it)    = e(it) - a * e(it-1);
%         e(it)   = e(it) - sum(be(1:it-1).*e(it-1:-1:1));
end
