function [Phi,Q,Qx,Qy,Psi]=fdm2a(gr,Kx,Ky,IBOUND,Phi,Q,varargin)
%FDM2 a 2D block-centred steady-state finite difference model
% USAGE:
%    [Phi,Q,Qx,Qy,Psi]=fdm2(gr,Kx,Ky,IBOUND,FH,FQ[,'rch',rch][,'drn',drn][,'ghb',ghb]...
%        [,'AXIAL',true|false][,'LAYCON',true|false][,LAYCON2',true|false])
% Inputs:
%    arguments between [ ] are optional
%    gr    = grid2DObj (see grid2DObj)
%    Kx,Ky = conductivities. If no Z is used when specifying the grid,
%            then treat them as transmissivities because default dZ=1.
%    IBOUND= boudary array as in MODFLOW (<0 fixed head, 0 inactive, >0 active)
%    IH    = initial heads (STRTHD in MODFLOW)
%    FQ    = fixed (prescribed) Q for each cell.
%    rch   = array (Ny,Nx) with recharge rate values
%    ghb   = cell or list of cells with {h C}  h = head
%    drn   = cell or list of cells with {h C}  h = elevation of drain
%    'AXIAL' , true|false --- grid is axially symmetric (y upward)
%    'LAYCON', true|false --- for horizontal model : thickness Z of grid will be adapated to head
%    'LAYCON2',true|false --- for vertial    model : thickness gr.dY willbe adapated to head
% Outputs:
%    Phi,Q computed heads and cell balances
%    Qx(Ny,Nx-1) horizontal cell face flow positive in positive xGr direction
%    Qy(Ny-1,Nx) vertial    cell face flow, postive in positive yGr direction
%    Psi(Ny+1,Nx-2)  stream function (only useful if flow is divergence free)
% TO 991017  TO 000530 001026 070414 090314 101130 140410
global Ifh Inact Iact iLoop

[ghb,    varargin] = getProp(varargin,'ghb',[]); ghbs = ~isempty(ghb);
[drn,    varargin] = getProp(varargin,'drn',[]); drns = ~isempty(drn);
[riv,    varargin] = getProp(varargin,'riv',[]); rivs = ~isempty(riv);
[rch,    varargin] = getProp(varargin,'rch',[]); rchs = ~isempty(rch);
[~  ,    varargin] = getProp(varargin,'chd',[]); chds = ~isempty(rch);

[TOL, ~          ] = getProp(varargin,'tol',1e-4');

Nodes = reshape(1:gr.Nod,gr.size);               % Node numbering
IE=Nodes(:,2:end);   IW=Nodes(:,1:end-1);
IS=Nodes(2:end,:);   IN=Nodes(1:end-1,:);

Iact  = IBOUND(:) >0;
Inact = IBOUND(:)==0;
Ifh   = IBOUND(:) <0;

Phi(Inact)=NaN; Q(Inact)=NaN;

if ghbs
    GHB = cell2list(ghb);
    Cghb = zeros(gr.Nod,1);
    hghb = zeros(gr.Nod,1);

    Cghb(GHB(:,end-2))=GHB(:,end);
    hghb(GHB(:,end-2))=GHB(:,end-1);
end

if drns
    Cdrn = zeros(gr.Nod,1); 
    hdrn = zeros(gr.Nod,1);

    DRN = cell2list(drn);
    Cdrn(DRN(:,end-2))=DRN(:,end);
    hdrn(DRN(:,end-2))=DRN(:,end-1);
    bdrn = Phi(:)>=hdrn(:);
end

if rivs
    Criv = zeros(gr.Nod,1); 
    hriv = zeros(gr.Nod,1);
    hbot = zeros(gr.Node,1);

    RIV = cell2list(riv);
    Criv(RIV(:,end-3))=RIV(:,end-1);
    hriv(RIV(:,end-3))=RIV(:,end-2);
    hbot(RIV(:,end-3))=RIV(:,end);
    briv = Phi(:)>=hbot(:);
end

if rchs,   Q = Q + gr.Area.*rch;   end

Q = Q(:); Phi=Phi(:); phiOld = Phi;

if gr.AXIAL, gr.LAYCON = 0; end  % safety


if gr.LAYCON || gr.LAYCON2 || drns || rivs
    MAXITER=50;
else
    MAXITER= 1;
end

fprintf('LAYCON = %d, LAYCON2 = %d, fxhds=%d drns= %d,  rivs=%d,  ghbs=%d, chds=%d  MAXITER = %d\n',...
    gr.LAYCON, gr.LAYCON2, any(IBOUND(:)<0), drns, rivs, ghbs, chds, MAXITER);

for iLoop=1:MAXITER
    
    fprintf('.'); if rem(iLoop,50)==0, fprintf('\n'); end

    if iLoop==1 || gr.LAYCON || gr.LAYCON2
        [Cx,Cy] = conductances(gr,Kx,Ky,Phi,IBOUND);

        C     = sparse([IW(:); IN(:)], [IE(:); IS(:)], [-Cx(:); -Cy(:)], gr.Nod, gr.Nod, 5*gr.Nod);
        C     = C + C';
        diagC = -sum(C,2);  % Main diagonal
    end
    
    D      = diagC;
    RHS    = Q-C(:,Ifh)*Phi(Ifh);

    if ghbs, D = D +       Cghb; RHS = RHS +       Cghb.*hghb; end
    if drns, D = D + bdrn.*Cdrn; RHS = RHS + bdrn.*Cdrn.*hdrn; end
    if rivs, D = D + briv.*Criv; RHS = RHS +       Criv.*(hriv-(1-briv).*hbot); end

    Phi(Iact) = spdiags(D(Iact),0,C(Iact,Iact))\RHS(Iact); % solve

    if drns
        bdrn(Iact) = Phi(Iact)>hdrn(Iact);
    end
    if rivs
        briv(Iact) = Phi(Iact)>hbot(Iact);
    end
    
    if all(abs(Phi(Iact)-phiOld(Iact))<TOL)
        fprintf('converged!\n');
        break;
    end

    phiOld(Iact) = Phi(Iact);

end
fprintf('\n');

Q(~Inact) = spdiags(diagC(~Inact),0,C(~Inact,~Inact))* Phi(~Inact);		% reshape

Phi = reshape(Phi,gr.size);
Q   = reshape(Q  ,gr.size);

Qx = -Cx.*diff(Phi,1,2);   % Flow across vertical   cell faces
Qy = +Cy.*diff(Phi,1,1);   % Flow across horizontal cell faces

Psi= [flipud(cumsum(flipud(Qx)));zeros(size(Qx(1,:)))];

if gr.LAYCON2
    Phi(Phi<gr.yGr(2:end)*ones(1,gr.Nx))=NaN;
    Psi(Phi<gr.yGr(2:end)*ones(1,gr.Nx))=NaN;
end

function [Cx,Cy] = conductances(gr,Kx,Ky,Phi,IBOUND)
    %CONDUCTANCE --- compute conductances
    % USAGE: [Cx,Cy] = contuctance(gr,Kx,Ky)
    global Ifh Inact Iact iLoop
    
    DY  = gr.dY;

    if gr.LAYCON2 && iLoop>1
        Phi = reshape(Phi,gr.size);
        zBot= gr.yGr(2:end) * ones(1,gr.Nx);        
        Inact = IBOUND==0 | Phi <= zBot;
        Iact  = IBOUND >0 & Phi >  zBot;
        Ifh   = IBOUND <0;
        DY = max(min(Phi-zBot,gr.dY),0.001);
    end
    Kx(Inact) = 0;
    Ky(Inact) = 0;

    % resistances and conducctances
    
    if ~gr.AXIAL
        if gr.LAYCON>0  % phreatic, adjust layer thickness
            Z1 = gr.Z(:,:,1); Phi(Phi>Z1(:)) = Z1(Phi>Z1(:));
            Z2 = gr.Z(:,:,2); Phi(Phi<Z2(:)) = Z2(Phi<Z2(:));
            H = reshape(Phi - Z2(:),gr.size);
        else
            H = gr.dZ;
        end
        
        RX = 0.5*bsxfun(@rdivide,gr.dx,DY)./(Kx.*H);
        RY = 0.5*bsxfun(@rdivide,DY,gr.dx)./(Ky.*H);
        Cx = 1./(RX(:,1:end-1)+RX(:,2:end));
    else
        RX = bsxfun(@rdivide,log(gr.xGr(2:end-1)./gr.xm( 1:end-1)),2*pi*Kx(:,1:end-1).*DY(:,1:end-1))+ ...
             bsxfun(@rdivide,log(gr.xm( 2:end  )./gr.xGr(2:end-1)),2*pi*Kx(:,2:end  ).*DY(:,2:end  ));
        RY = 0.5*bsxfun(@rdivide,gr.dY./Ky, pi*(gr.xGr(2:end).^2 - gr.xGr(1:end-1).^2));
        Cx = 1./RX;
    end
    Cy = 1./(RY(1:end-1,:)+RY(2:end,:));
    
    Inact = Inact(:); Iact=Iact(:); Ifh=Ifh(:);

