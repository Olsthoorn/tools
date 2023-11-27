function P = Moc(gr,Qx,Qy,Peff,R,t,varargin)
% MOC (massive particle tracking, steady state) with random walk and decay
%
% USAGE P = Moc(gr,Qx,Qy,Peff,R,t,n    ,  'aL',aL,'aT',aT,'Diff',Diff,'lambda',lambda);
%       P = Moc(gr,Qx,Qy,Peff,R,t,xP,yP,  'aL',aL,'aT',aT,'Diff',Diff,'lambda',lambda);
%
%  gr = grid2DObj
%  Qx,Qy from [Phi,Q,Qx,Qy] = Fdm2(...)
%  Peff = effective porosity
%  R    = [ - ] retardation, default 1
%  t    = [ T ] times to compute particle position at
%  n    = [   ] number of equally distributed particles per cell
%  xP,yP= [ L ]swarm of particles (real world coordinates)
%  rest of the input parameters is optional
%  aL   = [ L ] longitudinal dispersivity, default []
%  aT   = [ L ] transversal  dispersivity, default []
%  Diff = [L2/T] diffusion coefficient,     default []
%  aL, aT and Diff may be scalar or full cell arrays
%  lambda = decay parameter [1/T], default [];
%  randWalk is only initialize diff all aL, aT and Diff are given.
%  P struct contains P.time, P.x and P.y for all times
%
% TO 140418

% Values necessary to update cell when particle moves into adjacent cell
RWALL= [  -0.5,  +0.5, -0.5,+0.5]; % cell boundaries in relative coordinates
DIDX = [-gr.Ny, gr.Ny, -1  , 1  ]; % Idx to change if moving to neighbor cell
XRNW  = [  0.5,  -0.5,  0  , 0  ]; % Add to create new relative x coordinate
YRNW  = [  0  ,   0  ,  0.5,-0.5]; % Add to create new relative y coordinate

%% Get data for dispersion and diffusion
[aT  ,varargin]   = getProp(varargin,'aT',[]);
[aL  ,varargin]   = getProp(varargin,'aL',[]);
[Diff,varargin]   = getProp(varargin,'Diff',[]);
[lambda,varargin] = getProp(varargin,'lambda',[]);

if isscalar(R), R = gr.const(R); end

decay        = ~isempty(lambda);

if decay
    if isscalar(lambda), lambda = gr.const(lambda); end
end

mustRandWalk = ~isempty(aT) || ~isempty(Diff);
if mustRandWalk
    if isempty(aL),    aL   = 0;     end
    if isempty(aT),    aT   = aL/10; end
    if isempty(Diff),  Diff = 0;     end

    if isscalar(aL),   aL   = gr.const(abs(aL)  ); end
    if isscalar(aT),   aT   = gr.const(abs(aT)  ); end
    if isscalar(Diff), Diff = gr.const(abs(Diff)); end
end

%% Initials, constant when steady state
dt     = diff(unique([0; t(:)]));
epsVR  = gr.Vol.*Peff.*R;

% Zero flow at extends of model (always) [L3/T]
Qx = [zeros(gr.Ny,1), Qx, zeros(gr.Ny,1)];  
Qy = [zeros(1,gr.Nx); Qy; zeros(1,gr.Nx)];

activeCells = ~reshape( Qx(:,1:end-1)>=0 & Qx(:,2:end)<=0 & Qy(1:end-1,:)<=0 & Qy(2:end,:)>=0, [gr.Nod,1]);

% Accelleration [1/T]
ax  =  diff(Qx,1,2)./epsVR;
ay  = -diff(Qy,1,1)./epsVR;

% Velocity in cell centers [1/T]
vx =  0.5*(Qx(:,1:end-1)+Qx(:,2:end))./epsVR;
vy = -0.5*(Qy(1:end-1,:)+Qy(2:end,:))./epsVR;


%% Initialize particles
if numel(varargin)<2
    n = varargin{1};
    [xr0,yr0,Icells,Np] = distrParticles(gr,n);
else    
    [xr0,yr0,Icells,Np] = prepSwarm(gr,varargin{1},varargin{2});
end

xr = xr0;
yr = yr0;

%% Debug
xrOld = xr;
yrOld = yr;
IcellsOld = Icells;

%%
x  = reshape(gr.Xm(Icells),size(Icells)) + reshape(gr.dX(Icells),size(Icells)).*xr0;
y  = reshape(gr.Ym(Icells),size(Icells)) - reshape(gr.dY(Icells),size(Icells)).*yr0;

active = activeCells(Icells);

%% Initialyze time steps
DtRemaining = zeros(Np,4); % DtRemaining computed for each of the 4 directions
DtRest      = zeros(Np,1); % Remaining time in current time step dt(it-1)
tStep       = zeros(Np,1); % Intermediate time step (within step)

%% Initialize output struct
for it=numel(t):-1:1
    P(it).x      = zeros(size(x));
    P(it).y      = zeros(size(x));
    P(it).active = true( size(x));
    P(it).Icells = zeros(size(x));
    if decay
        P(it).mass = zeros(size(x));
    end
end
P(1).time   = t(1);
P(1).x(:)      = x;
P(1).y(:)      = y;
if decay
    P(1).mass(:) = 1;
end
P(1).active(:) = active;
P(1).Icells(:) = Icells;

fprintf('Simulating tracking of %d particles over %d time steps\n.',Np,numel(dt));

%% Loop over times, starting at time 2
for it=2:numel(t)
    
    fprintf('.');  if rem(it,50)==0, fprintf('%d\n',it); end

    % initialize next time step
    DtRest(:) = dt(it-1);
        
    Ip = find(DtRest>0 & active);  % Indices of remaining points        
    
    while ~isempty(Ip) % loop until DtRest == 0 for all particles
        
        K  = zeros(size(Ip));
        Ic = Icells(    Ip);
        if max(Ic)>gr.Nod
            error('max Ic>gr.Nod');
        end
        % Forward computation of particles position
        xr(Ip) = estimate(Ip,Ic,ax(:),vx(:),DtRest,xr0);
        yr(Ip) = estimate(Ip,Ic,ay(:),vy(:),DtRest,yr0);
        
        % particles that passed one or more cell faces
        L{1}  = xr(Ip)<=RWALL(1);
        L{2}  = xr(Ip)>=RWALL(2);
        L{3}  = yr(Ip)<=RWALL(3);
        L{4}  = yr(Ip)>=RWALL(4);

        % particles that are still within their original cell
        L0    = ~(L{1} | L{2} | L{3} | L{4});
        
        % all points that not hitting a cell face are done in this loop
        xr0(   Ip(L0)) = xr(Ip(L0));
        yr0(   Ip(L0)) = yr(Ip(L0));
        DtRest(Ip(L0)) = 0;

        % ===== All points that hit any cell face (~L0) =====================

        % Compute time till particle hits a cell face
        DtRemaining(Ip,:) = 0;
        
        for i=1:2  % West and East
            if any(L{i})
                Ipp = Ip(L{i});
                Icc = Icells(Ipp);            
                DtRemaining(Ipp,i) = estimateTrest(Ipp,Icc,ax(:),vx(:),DtRest,xr0,RWALL(i));
            end
        end

        for i=3:4 % North and South
            if any(L{i})
                Ipp = Ip(L{i});
                Icc = Icells(Ipp);
                DtRemaining(Ipp,i) = estimateTrest(Ipp,Icc,ay(:),vy(:),DtRest,yr0,RWALL(i));
            end
        end

        [dt_remaining,kk] = max(DtRemaining(Ip(~L0),:),[],2);

        kk(dt_remaining==0)=0;
        
        K(~L0) = kk;
        
        if ~isempty(Ip(~L0))
             tStep(Ip(~L0)) = DtRest(Ip(~L0)) - dt_remaining;
            DtRest(Ip(~L0)) = dt_remaining;
        end
        
        % if K==0, i.e. point at cell face,but dt_remaining = 0
        % then move point to adjacent cell
        
        for i=4:-1:1
            Ipp = Ip(L{i} & (K==0));
            if any(Ipp)
                Icells(Ipp) = Icells(Ipp) + DIDX(i);
                if i <=2,  xr(Ipp)= XRNW(i); xr0(Ipp) = XRNW(i); end
                if i >=3,  yr(Ipp)= YRNW(i); yr0(Ipp) = YRNW(i); end
            end
            Ipp = Ip(L{i} & K==i);
            if any(Ipp)
                Icc= Icells(Ipp);
                
                xrOld(Ipp)     = xr(Ipp);
                yrOld(Ipp)     = yr(Ipp);
                IcellsOld(Ipp) = Icells(Ipp);
                
                xr(Ipp) = estimate(Ipp,Icc,ax(:),vx(:),tStep,xr0);
                yr(Ipp) = estimate(Ipp,Icc,ay(:),vy(:),tStep,yr0);
                
                Icells(Ipp) = Icells(Ipp) + DIDX(i);
                if any(Icells(Ipp)>gr.Nod)
                    iip = Ipp(Icells(Ipp)>gr.Nod);
                    iic = IcellsOld(iip);
                    iicN= Icells(iip)
                    xOld = gr.Xm(iic)+gr.dX(iic);
                    error('stop');
                end
                
                if i <=2,  xr(Ipp)= XRNW(i); xr0(Ipp) = XRNW(i); end
                if i >=3,  yr(Ipp)= YRNW(i); yr0(Ipp) = YRNW(i); end
            end
        end
        
        Ip = Ip(DtRest(Ip)>0);
    end
    
    active = activeCells(Icells);
    
    xr(~active) = 0;       % put particles in sinks in
    yr(~active) = 0;       % center of sink

    x  = reshape(gr.Xm(Icells),size(Icells)) + xr.*reshape(gr.dX(Icells),size(Icells));
    y  = reshape(gr.Ym(Icells),size(Icells)) - yr.*reshape(gr.dY(Icells),size(Icells));

    Ip = active;
    Ic = Icells(active);

    if mustRandWalk && ~isempty(Ic)
        % dx and dy are based solely on the velocity at the end points
        [dxRW,dyRW] = ...
            randWalk(gr,Ip,Ic,xr,yr,ax(:),ay(:),vx(:),vy(:),aL(:),aT(:),Diff(:),R(:),dt(it-1));
          x(Ip) = x(Ip) + dxRW;
          y(Ip) = y(Ip) - dyRW;

          % mirror particles that jumped over the model boundary
          I = x<gr.xGr(  1); x(I) = 2*gr.xGr(  1)-x(I);
          I = x>gr.xGr(end); x(I) = 2*gr.xGr(end)-x(I);
          I = y>gr.yGr(  1); y(I) = 2*gr.yGr(  1)-y(I);
          I = y<gr.yGr(end); y(I) = 2*gr.yGr(end)-y(I);

          if any(isnan(Icells)), error('stop'); end
         [Icells(Ip),~,~,~,~,xr(Ip),yr(Ip)] = gr.Idx(x(Ip),y(Ip));        
         active(Ip) = activeCells(Icells(Ip));
    end
    
    % store output in struct
    P(it).time   = t(it);     %#ok
    P(it).x(:)      = x;      %#ok
    P(it).y(:)      = y;      %#ok
    P(it).active(:) = active; %#ok
    P(it).Icells(:) = Icells; %#ok
    
    if decay
        P(it).mass      = P(it-1).mass .* exp(-reshape(lambda(Icells),size(Icells)).*dt(it-1)); %#ok
    end
    
    % update which particles are active
    xr0(active) = xr(active);
    yr0(active) = yr(active);

end
fprintf('%d done\n',it);

%% Functions used above

function [xr0,yr0,Icells,Np] = distrParticles(gr,n)
    %distrParticles -- generate n by n particles in all cells
    % USAGE: [xr0,yr0,Icells,Np] = distrParticles(gr,n)
    %  if n<0, particles are randomized by adding a random number between
    %  0 and 1 ot their relative cell coordinate and mirroring it at their
    %  cell face to keep the particles in their origial cells.
    %  TO 140418
    randomize = n<0;
    n = abs(n);
    u = (1/(2*n):1/n:1-1/(2*n))-0.5;
    for i=n:-1:1
        uu = u(i);
        for j=n:-1:1
            IPnt =  (gr.Nod*(n*(i-1)+j-1) +1):gr.Nod*(n*(i-1)+j);
            vv = u(j);
            xr0(IPnt,1)  = uu;
            yr0(IPnt,1)  = vv;
            Icells(IPnt,1)  = 1:gr.Nod;  % ix of these points
           % active(IPnt,1)  = activeCells;
        end
    end
    Np = gr.Nod * n*n;     % Total number of particles
    
    if randomize
        xr0 = xr0 + rand(size(xr0))-0.5;
        yr0 = yr0 + rand(size(yr0))-0.5;
        
        %mirrors at cell faces to keep particles in their original cells
        xr0(xr0<-0.5) = -1 - xr0(xr0<-0.5);
        xr0(xr0> 0.5) =  1 - xr0(xr0> 0.5);
        yr0(yr0<-0.5) = -1 - yr0(yr0<-0.5);
        yr0(yr0> 0.5) =  1 - yr0(yr0> 0.5);
    end

function [xr0,yr0,Icells,Np] = prepSwarm(gr,xP,yP)
    %prepSwarm -- prepares a swarm of particles
    % USAGE: [xr0,yr0,Icells,Np] = prepareSwarm(gr,xP,yP)  
   [Icells,~,~,~,~,xr0,yr0] = gr.Idx(xP(:),yP(:));
    L      = ~isnan(Icells);
    xr0    = xr0(L);
    yr0    = yr0(L);
    Icells = Icells(L);
    Np     = numel(Icells); 

function r = estimate(Ip,Ic,a,v,Dt,r0)
    %ESTIMATE -- compute new relative coordinate (x or y direction)
    %USAGE: xr = estimate(Ip,Ic,ax,vx,Dt,xr0)
    % r, r0 = relative coordinate
    
    T_MAX = 1e6;
    
    r = zeros(size(Ip));

    L = abs(a(Ic))>1/T_MAX;
    
    if any(~L)
        r(~L) =  v(Ic(~L)).*Dt(Ip(~L)) +r0(Ip(~L));
    end
    r( L) = (v(Ic( L)) + a(Ic( L)).*r0(Ip( L))).*(exp(a(Ic(L)).*Dt(Ip(L))) - 1)./a(Ic(L)) + r0(Ip(L));

function tRemaining = estimateTrest(Ip,Ic,a,v,DtRest,r0,RWALL)
    %ESTIMATETREST -- compute remaining time given RWALL and DtRest
    %USAGE: tRemaining = estimateTrest(Ip,Ic,a,v,DtRest,r0,RWALL)
    T_MIN = 1e-3; % about a minute;
    T_MAX = 1e6;  % about 1000 days
    tRemaining = zeros(size(Ip));

    % Note that dt = log(v1/v0)/a) and that v1/v0 >0 is required to yield a
    % positive dt. In the limit this log becomes
    % dt = log(dV/V0-1)/a = dV/(aV0)
    % which equals dt =(r-r0)/(V0+r0a)
    % so this is the correct time check to judge whether the log gives a
    % postive time and when r0a matters, it does when T_MAX>1/a, i.e.
    % then a>1/T_MAX
    
    La   = a(Ic)>1/T_MAX & (RWALL-r0(Ip))./(v(Ic)+r0(Ip).*a(Ic)) > 0; %T_MIN;
    Lv   = ~La           & (RWALL-r0(Ip))./ v(Ic)                > 0; %T_MIN;

    % Handle points in cells with abs(a)>EPS first, i.e. use log
    arg = (v(Ic(La)) + a(Ic(La)) .* RWALL)./ ...
          (v(Ic(La)) + a(Ic(La)) .* r0(Ip(La)));
    tRemaining(La) = DtRest(Ip(La)) - log(arg)./a(Ic(La));

    % Handle point in cells with abs(a)<EPS second, i.e. constant velocity
    tRemaining(Lv) = DtRest(Ip(Lv)) - (RWALL - r0(Ip(Lv)))./v(Ic(Lv));
    
    % assembling
    tRemaining(tRemaining <             T_MIN) = 0;
    tRemaining(tRemaining > DtRest(Ip)- T_MIN) = 0;
    
function [dx,dy] = randWalk(gr,Ip,Ic,xr,yr,ax,ay,vx,vy,aL,aT,Diff,R,dt)
    %randWalk -- take a random walk stap
    % USAGE [xr,yr] = randomWalk(xr,yr,ax,ay,vx,vy,aL,aT,Diff,dt)
    VX     = (vx(Ic) + ax(Ic) .* xr(Ip)).*reshape(gr.dX(Ic),size(Ic)); % true VX
    VY     = (vy(Ic) + ay(Ic) .* yr(Ip)).*reshape(gr.dY(Ic),size(Ic)); % true VY
    
    V      = sqrt(VX.^2 + VY.^2); % absolute velocity
    
    sigmaL = sqrt(2 * (aL(Ic).*V+Diff(Ic))./R(Ic) * dt); % logitudinal
    sigmaT = sqrt(2 * (aT(Ic).*V+Diff(Ic))./R(Ic) * dt); % transverse
    
    dsLong = randn(numel(xr(Ip)),1).*sigmaL; % random longitudinal displacement
    
    L          = dsLong < -V*dt./R(Ic);
    dsLong (L) = -V(L)*dt./R(Ic(L));    % upstream dispersion is physically 0
    
    dsTrans= randn(numel(xr(Ip)),1).*sigmaT; % transverse displacement
    dx     = (VX.*dsLong -VY.*dsTrans)./V;   % total in x-direction
    dy     = (VY.*dsLong +VX.*dsTrans)./V;   % total in y-direction
    

