function [Phi,Q,Qx,Qy,Psi]=fdm2(gr,Tx,Ty,IBOUND,Phi,Q)
%FDM2 a 2D block-centred steady-state finite difference model
% USAGE:
%    [Phi,Q,Qx,Qy,Psi]=fdm2(gr,Tx,Ty,IBOUND,FH,FQ)
% Inputs:
%    gr    = grid2DObj (see grid2DObj)
%    Tx,Ty = transmissivities, either scalar or full 2D arrays
%    IBOUND= boudary array as in MODFLOW (<0 fixed head, 0 inactive, >0 active)
%    IH    = initial heads (STRTHD in MODFLOW)
%    FQ    = fixed (prescribed) Q for each cell.
% Outputs:
%    Phi,Q computed heads and cell balances
%    Qx(Ny,Nx-1) horizontal cell face flow positive in positive xGr direction
%    Qy(Ny-1,Nx) vertial    cell face flow, postive in positive yGr direction
%    Psi(Ny+1,Nx-2)  stream function (only useful if flow is divergence free)
% TO 991017  TO 000530 001026 070414 090314 101130 140410

Nodes = reshape(1:gr.Nod,gr.size);               % Node numbering
IE=Nodes(:,2:end);   IW=Nodes(:,1:end-1);
IS=Nodes(2:end,:);   IN=Nodes(1:end-1,:);

Iact  = IBOUND(:) >0;
Inact = IBOUND(:)==0; Tx(Inact)=0; Ty(Inact)=0;
Ifh   = IBOUND(:) <0;

% resistances and conducctances
RX = 0.5*bsxfun(@rdivide,gr.dx,gr.dy)./Tx;
RY = 0.5*bsxfun(@rdivide,gr.dy,gr.dx)./Ty;

Cx = 1./(RX(:,1:end-1)+RX(:,2:end));
Cy = 1./(RY(1:end-1,:)+RY(2:end,:));

C     = sparse([IW(:); IN(:)], [IE(:); IS(:)], [-Cx(:); -Cy(:)], gr.Nod, gr.Nod, 5*gr.Nod);
C     = C + C';
diagC = -sum(C,2);  % Main diagonal

Phi = Phi(:); Phi(Inact)=NaN;
Q   = Q(  :); Q(  Inact)=NaN;

Phi(Iact) = spdiags(diagC(  Iact),0,C( Iact , Iact ))\(Q(Iact)-C(Iact,Ifh)*Phi(Ifh)); % solve
Q(~Inact) = spdiags(diagC(~Inact),0,C(~Inact,~Inact))* Phi(~Inact);		% reshape

Phi = reshape(Phi,gr.size);
Q   = reshape(Q  ,gr.size);

Qx = -Cx.*diff(Phi,1,2);   % Flow across vertical   cell faces
Qy = +Cy.*diff(Phi,1,1);   % Flow across horizontal cell faces

Psi= [flipud(cumsum(flipud(Qx)));zeros(size(Qx(1,:)))];
