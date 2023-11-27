function ds=PartialPenetration(Q,kD,rz,Ztop,Zbot,ZScrT,ZScrB,kroverkz)
%PARTIALPENETRATION -- computes extra drawdown ds in point r,z due to
%partialpenetration (see Kruseman & De Ridder)
%
% USAGE
%    ds=partialpenetration(Q,kD,r|rz,[z],Ztop,Zbot,ZScrT,ZscrB[,kroverkz])
%
% Inputs
%    Q     = extracttion (postitive yields positive pp)
%    kD    = transmissivity
%    rz    = coordinates of points size(nPoitns,2) for which pp is desired
%    Ztop  = elevation of aquifer top
%    Zbot  = elevation of aquifer bottom
%    ZScrT = elevation of top of screen
%    ZScrB = elevation of bottom of screen
%    kroverkz optional vertical anisotropy which is kr/kz
%
% Output:
%       ds(size(rz)) = partial penetration
%
% interchanging the direction of z is allowed. If z downward all distrances
% are from bottom of the aquifer and tops and bottoms are interchanged
%
% TO 090330 151214

epsStop = 1e-5;

%% Anisotropy
kD    =kD*kroverkz^(1/3);
ani_z =kroverkz^( 2/6);
ani_r =kroverkz^(-1/6);
Zbot  =Zbot*ani_z;
Ztop  =Ztop*ani_z;
ZScrT =ZScrT*ani_z;
ZScrB =ZScrB*ani_z;
r     =rz(:,1)*ani_r;
z     =rz(:,2)*ani_z;

%% Set fixed values
D = abs(Ztop- Zbot );
d = abs(ZScrT-ZScrB);

%% Compute series, for all points simultaneously

ds=zeros(size(r));
for n=1:500  % analytical solution partial penetration
    K0  = besselk(0,n*pi.*r/D);
    ds =  ds + 1/n*(sin(n*pi*(ZScrB-Zbot)/D)-sin(n*pi*(ZScrT-Zbot)/D))...
                 .* cos(n*pi.*(z-Zbot)/D)...
                 .* K0;
         
         if all(K0<epsStop)
             break;
         end
end

ds=-ds*Q/(2*pi*kD)*(2*D)/(pi*d);
    