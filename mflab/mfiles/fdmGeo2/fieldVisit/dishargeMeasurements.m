%% Field visit May 6 with studentes TU CIE5440, geohydrology 2
%% With follow up on May 7, by Pierre Kamps and Theo Olsthoorn
% TO 140507

%% Discharge measurments

g = 9.81; % m2/s
secPerDay = 86400;
format1 = '%-25s, %6.0f m3/d\n';
format2 = '%-25s, z=%6.2f h=%6.2f d=%6.2f W=%6.2f Q=[%6.0f,%6.0f, %6.0f] m3/d\n';

Qm3pd  = @(z,h,d,W) h * sqrt(g/2/z) * d * W * secPerDay;

%% New flat steel weir in 1000 m canal
z = 0.49; % fall height
h = 0.25; % distance of wateterfall forward
d = 0.04; % water layer thickness on flat weir
W = 2.10; % weir width
L = 0.15; % crest width

[Qhs,Qd] = Qhsd(d,W,L);
fprintf(format2,'Weir, May 6',z,h,d,W,Qm3pd(z,h,d,W),Qhs,Qd);
%% Weir U5
z = 0.39; h = 0.35; d = 0.09; W = 0.78;  L=0;

[Qhs,Qd] = Qhsd(d,W,L);
fprintf(format2,'Outlet U5, May 6',z,h,d,W,Qm3pd(z,h,d,W),Qhs,Qd);

%% Weir U4
z = 0.818-0.32; h = 0.32; d = 0.095; W = 0.76; L=0;
[Qhs,Qd] = Qhsd(d,W,L);
fprintf(format2,'Outlet U4, May 6',z,h,d,W,Qm3pd(z,h,d,W),Qhs,Qd);

%% May 07 14:00-16:00h Pierre - Theo


% Weir overlet from pond 6 to pond 7 (supply of pond 7)
z = 0.86-0.515; h = 0.27; d = 0.088; W = 0.50; L=0.06;
[Qhs,Qd] = Qhsd(d,W,L);
fprintf(format2,'Inlet pond 7, May 7',z,h,d,W,Qm3pd(z,h,d,W),Qhs,Qd);

%% U5 overlet
z = 0.59-0.17; h = 0.36; d = 0.100; W = 0.75; L=0.0;
[Qhs,Qd] = Qhsd(d,W,L);
fprintf(format2,'Outlet U5, May 7',z,h,d,W,Qm3pd(z,h,d,W),Qhs,Qd);

%% Trial to measure discharge U5
% By the decline of the water table in the weir chamber from
% where v=0 to top of crest 25.8 where v = maximum 27.5

% dh   = 0.017; %m
% v    = sqrt(2 * g * dh);
% fprintf(format1,'Outlet U5, May 7, v=sqrt(2gdh)',v*d*W*secPerDay,Qam3pd(d,W));
