function eMeasNew(Q,r,T,S)
%% genereate data for modelScript5 for testing simple calibration using
% lsqnonlin

tMin  = [1 2 3 5 7 10 15 20 30 40 50 60 75 90 105 120 150 180:90:600];
tDays = tMin/(24*60);

T = T*exp(0.25*randn);
S = S*exp(0.25*randn);

Theis = @(t) Q/(4*pi*T) * expint(r^2*S./(4*T*t));


s = Theis(tDays);

err = 0.05 * s(:) .* randn(size(s(:))); % error to be added

eMeas = [tDays(:) s(:) + err(:)];

save eMeas eMeas