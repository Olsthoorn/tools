function [e,br,be,yHat]= FunGamma(p,t,X,Y)
%Fun -- convolution for lsqnonlin using analytical block response
% theta = A/t * exp(-alpha^2/(beta^2*t)-beta^2*t)
% y = A + w
% A = conv(Fun,p)
% w = conv(exp,a)
dt = mean(diff(t));

a = exp(p(1));    % exponential decay parameter noise model
A = exp(p(2));    % gain
a2= exp(p(3));
n = exp(p(4));    % parameter in br
d = exp(p(5));
fprintf('a = %g, A=%g a2=%g n=%g d=%g\n',a,A,a2,n,d);

% IR
t = t - t(1)+dt;
br = A.*a2^n.*exp(-a2*t)/gamma(n);
be = sqrt(2*a) * exp(-a * (1:numel(t))*dt)';

%yHat = zeros(size(t));
yHat = filter(br,1,X)+d;
e    = Y-yHat;
for it=2:numel(t)
%      yHat(it) = sum(br(1:it).*X(it:-1:1))+ a * e(it-1) + d; % sum(be(1:it-1).*e(it-1:-1:1)) + d;
%      e(it)    = e(it) - a * e(it-1);
       e(it)    = e(it) - sum(be(1:it-1).* e(it-1:-1:1));
end
