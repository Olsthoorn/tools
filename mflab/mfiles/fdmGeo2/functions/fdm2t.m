function [Phi,Q]=fdm2t(gr,t,Tx,Ty,Ss,IBOUND,HI,QI)
%FDM2 a 2D block-centred transient finite difference model
% USAGE:
%    [Phi,Q]=fdm2(gr,t,Tx,Ty,Ss,IBOUND,HI,FQ)
% Inputs:
%    gr    = grid2DObj (see grid2DObj)
%    Tx,Ty = transmissivities, either scalar or full 2D arrays
%    IBOUND= boudary array as in MODFLOW (<0 fixed head, 0 inactive, >0 active)
%    HI    = initial heads (STRTHD in MODFLOW)
%    QI    = prescribed inflow for each cell.
%    t     = times, dt = diff(t) will be the time steps
% Outputs:
%    Phi(Ny  ,Nx  ,Nt ) [ L  ]  Nt=numel(diff(t)); Ndt=numel(dt);
%    Qin(Ny  ,Nx  ,Ndt) [L3/T]  flow into cells during time step
%    Qx( Ny  ,Nx-1,Ndt) [L3/T]  horizontal cell face flow positive in positive xGr direction
%    Qy( Ny-1,Nx  ,Ndt) [L3/T]  vertial    cell face flow, postive in positive yGr direction
%    Psi(Ny+1,Nx-2,Ndt) [L3/T]  stream function (only useful if flow is divergence free)
%    Qs( Ny  ,Nx  ,Ndt) [L3/T]  flow released from storage during the time step Dphi*S*V/Dt
% TO 991017  TO 000530 001026 070414 090314 101130 140410

theta = 0.67; % degree of implicitness  

t     = permute(unique(t(:)),[3,2,1]); dt = diff(t,1,3); Ndt=numel(dt);
Nodes = reshape(1:gr.Nod,gr.size);               % Node numbering
IE=Nodes(:,2:end);   IW=Nodes(:,1:end-1);
IS=Nodes(2:end,:);   IN=Nodes(1:end-1,:);

Iact  = IBOUND(:) >0;
Inact = IBOUND(:)==0; Tx(Inact)=0; Ty(Inact)=0;
Ifh   = IBOUND(:) <0;

[Cx,Cy,Cs] = conductances(gr,Tx,Ty,Ss,theta);

C     = sparse([IW(:); IN(:)], [IE(:); IS(:)], [-Cx(:); -Cy(:)], gr.Nod, gr.Nod, 5*gr.Nod);
C     = C + C';     % add lower half of matrix
diagC = -sum(C,2);  % Main diagonal

HI(Inact)=NaN;   Phi = bsxfun(@times, t,HI);   HI = HI(:);   HT=HI;

QI(Inact)=  0;   QI = QI(:); % remains fixed
for idt = 1:Ndt
    HT(Iact) = spdiags(diagC(Iact)+Cs(Iact)/dt(idt),0,C( Iact , Iact )) ...
               \(QI(Iact) - C(Iact,Ifh)*HI(Ifh) + Cs(Iact)/dt(idt).*HI(Iact)); % solve
    
    Q(idt).NET= gr.const(0);
    Q(idt).STO= gr.const(0);
    Q(idt).FH = gr.const(0);
    
    % Water budget items for every cell
    % Q_NET is net inflow in cell to surrounding cells. In FH cells this
    % equals the flow Q_FH from fixed heads. See line 4 below Q_FH(Ifh)=Q_NET(Ifh).
    % in fact: Q_NET = Q_FH + Q_FQ = Q_STO
    Q(idt).NET(~Inact)  = reshape(spdiags(diagC(~Inact),0,C(~Inact,~Inact))* HT(~Inact),gr.size);
    Q(idt).STO(Iact)    = Cs(Iact)/dt(idt).*(HT(Iact)-HI(Iact));
    Q(idt).FH(Ifh)      = Q(idt).NET(Ifh);
    Q(idt).FQ           = QI;
    Q(idt).X            = -diff(reshape(HT,gr.size),1,2).*Cx;
    Q(idt).Y            =  diff(reshape(HT,gr.size),1,1).*Cy;

    % Stream function in XZ plane only useful in vertical cross sections
    Q(idt).Psi          = zeros(gr.Ny+1,gr.Nx-1);
    Q(idt).Psi(2:end,:) = cumsum(Q(idt).X,1);
    Q(idt).Psi(:,:)     = Q(idt).Psi(end:-1:1,:);
    
    % Update HI for next loop
    HI(Iact) = HT(Iact)/theta - (1-theta)/theta * HI(Iact);
    
    Phi(:,:,idt+1) = reshape(HI,gr.size);

end

function [Cx,Cy,Cs] = conductances(gr,Tx,Ty,Ss,theta)
    %CONDUCTANCE --- compute conductances
    % USAGE: [Cx,Cy] = contuctance(gr,Tx,Ty)
    
    if isscalar(Tx), Tx= gr.const(Tx); end
    if isscalar(Ty), Ty= gr.const(Ty); end
    if isscalar(Ss), Ss= gr.const(Ss); end
    
    % resistances and conducctances
    if ~gr.AXIAL
        RX = 0.5*bsxfun(@rdivide,gr.dx,gr.dy)./Tx;
        RY = 0.5*bsxfun(@rdivide,gr.dy,gr.dx)./Ty;
        Cx = 1./(RX(:,1:end-1)+RX(:,2:end));
        Cs = gr.Vol.*Ss/theta; Cs = Cs(:);
    else
        RX = bsxfun(@rdivide,log(gr.xGr(2:end-1)./gr.xm( 1:end-1)),2*pi*Tx(:,1:end-1).*gr.dY(:,1:end-1))+ ...
             bsxfun(@rdivide,log(gr.xm( 2:end  )./gr.xGr(2:end-1)),2*pi*Tx(:,2:end  ).*gr.dY(:,2:end  ));
        RY = 0.5*bsxfun(@rdivide,gr.dY./Ty, pi*(gr.xGr(2:end).^2 - gr.xGr(1:end-1).^2));
        Cs = gr.Vol.*Ss/theta; Cs = Cs(:);
        Cx = 1./RX;
    end
    Cy = 1./(RY(1:end-1,:)+RY(2:end,:));

