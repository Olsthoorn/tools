function e = FUN(p)
% used with simple calibration (modelScript5), called by lsqnonlin
global eMeas Q T0 S0 r

fprintf('.'); % shows each call by printing a dot

t = eMeas(:,1);

% Parameter update
T = exp(p(1)) * T0;
S = exp(p(2)) * S0;

% compute drawdown (would normally be hte model)
s = Q/(4*pi*T) * expint( r^2 *S./(4*T*t));

% output difference with measurements
e = s(:) - eMeas(:,2);
