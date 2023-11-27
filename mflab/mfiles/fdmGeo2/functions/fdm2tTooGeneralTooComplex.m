function [Phi,B]=fdm2t(varargin)
%FDM2 a 2D block-centred transient finite difference model
% USAGE:
%    [Phi,Q,Psi]=fdm2(gr,t,Kx,Ky,Ss,IBOUND,HI, ...
%            'wells',FQ,'ghb',GHB,'drn',DRN,'riv',RIV,'rch',RCH,'evt',EVT)
%
% Required inputs:
%    gr    = grid2DObj (see grid2DObj)
%    t     = times, dt = diff(t) will be the time steps
%    Kx,Ky = concuctivities, either scalar or full 2D arrays
%          use zGr in grid2DObj to set thickness of aquifer, Kx=Tx, Ky=Ty if
%          zGr is omitted in gridObj or thickness is set at 1 m.
%    Ss    = specific storage coefficient (note thickness of layer, through gr2DObj
%    IBOUND= boundary array as in MODFLOW (<0 fixed head, 0 inactive, >0 active)
%    HI    = initial heads (STRTHD in MODFLOW)
% Optional inputs:
%    all specified through parameterName/value pairs in any order
%    Inputs are in the form as required by mfLab:
%         'wells'  as {iSP iL iR iC Q}  or {iSP idx Q} or {iSP -1}
%         'GHB'    as {iSP iL iR iC HGHB CGHB} or {iSP idx HGHB CGHB}  or {iSP -1}
%         'DRN'    as {iSP iL iR iC HDRN CDRN} or {iSP idx HDRN CDRN}  or {iSP -1}
%         'RIV'    as {iSP iL iR iC HRIV CRIV HRIVBOT} or {iSP idx HRIV CRIV HRIVBOT}
%  input may also be given in list form, without subdivision in cells
%         'RCH'    as {iSP iL iR iC RCH} or {iSP idx RCH} or {iSP RCH} or {iSP -1}
%         'EVT'    as {iSP iL iR iC EVT} or {iSP idx EVT} or {iSP EVT} or {iSP -1}
%  input for RCH and EVT may also be given as a full 3D array where size(..,3) matches Nt
%  a single vector is possible as well (but watch out for models that have
%  the same dimension in x or y as Nt if a single vector is used.
%   A single scalar is considered as constant uniform value for all stress
% 
% Outputs:
%    Phi(Nt).values(Ny,Nx)  %  heads
%    B(it)     % cell by cell water budget during time step it
%         B(it).label{Nlbl} = labels nameing the cell by cell water budget type as in MODFLOW
%         B(it).term{Nlbl}(Ny,Nx,Nz)  = arrays of cell by cell budget of type
%                                  given in corresponding label as in MODFLOW
%    labels contain any or all of the following strings:
%        FLOWRIGHTFACE, FLOWFRONTFACE, FLOWLOWERFACE, CONSTANTHEAD, WELLS,
%        GHB, DRN, RIV, RCH, EVT, Psi
% Notice
%    FLOWRIGHTFACE has (Ny,Nx-1,Nz) values  % in 2D, Nz=1
%    FLOWFRONTFACE has (Ny-1,Nx,Nz) values  % in 2D, Nz-=1
%    FLOWLOWERFACE has (Ny,Nx,Nz-1) values  % in 2D, is empty
%    Psi(it)(Ny+1,Nx-1) [L3/T]  stream function (only useful if flow is divergence free)
%
% TO 991017  TO 000530 001026 070414 090314 101130 140410

    global HT HI C Cs diagC Cx Cy Iact Ifh Inact dt idt
    
    [gr,varargin]      = getType(varargin,'grid2DObj',[]);
    [time, varargin] = getNext(varargin,'double',[]);
    [Kx,   varargin] = getNext(varargin,'double',[]);
    [Ky,   varargin] = getNext(varargin,'double',[]);
    [Ss,   varargin] = getNext(varargin,'double',[]);
    [IBOUND,varargin]= getNext(varargin,'double',[]);
    [HI   ,varargin] = getNext(varargin,'double',[]);
    
    [bcn,theta] = parser(varargin{:});

    % Labels and stresses to include (depends on input)
    cbcAlways  = {'FLOWRIGHTFACE','FLOWFRONTFACE','FLOWLOWERFACE','STORAGE' 'CONSTANT HEAD'};
    stressLbl  = {'GHB','DRN','RIV','RCH','EVT','FQ'};
    for j=numel(stressLbl):-1:1
        if isempty(bcn.(stressLbl{j}))
            stressLbl(j)=[];
        end
    end
    
    cbcLbl = [cbcAlways stressLbl]; % labels cell by cell flows
    
    for j=1:numel(stressLbl)
        bcn.(stressLbl{j}) = checkInput(bcn.(stressLbl{j}));
    end

    time   = permute(unique(time(:)),[3,2,1]);
    dt     = diff(time,1,3);
    ndt    = numel(dt);

    Nodes  = reshape(1:gr.Nod,gr.size);               % Node numbering
    IE     = Nodes(:,2:end);   IW=Nodes(:,1:end-1);
    IS     = Nodes(2:end,:);   IN=Nodes(1:end-1,:);

    Iact   = IBOUND(:) >0;
    Inact  = IBOUND(:)==0; Kx(Inact)=0; Ky(Inact)=0;
    Ifh    = IBOUND(:) <0;

    [Cx,Cy,Cs] = conductances(gr,Kx,Ky,Ss,theta);

    C     = sparse([IW(:); IN(:)], [IE(:); IS(:)], [-Cx(:); -Cy(:)], gr.Nod, gr.Nod, 5*gr.Nod);
    C     = C + C';
    diagC = -sum(C,2);  % Main diagonal


    % allocate memory to store head and flows
    for idt=ndt:-1:1
        Phi(idt).values = gr.const(NaN);
        Phi(idt).time   = time(idt+1);
        B(  idt).label  = cbcLbl;
        B(  idt).time   = time(idt)+theta*dt(idt);
        for j=numel(cbcLbl):-1:1
            B(ndt).term{j} = gr.const(0);
        end
    end

    %% Main loop, solving once per time step

    HI (Inact) = NaN; HI = HI(:); HT = HI;

    for idt = 1:ndt

        % set diagonal D and RHS
        D         = diagC;
        RHS       = zeros(size(D));
        for j=1:numel(cbcLbl)
            [D,RHS] = setD_RHS(cbcLbl{j},D,RHS);
        end

        % solve    
        HT(Iact) = spdiags(D(Iact),0,C( Iact , Iact )) \RHS(Iact); % solve

        % compute head at end of time step
        HI(Iact) = HT(Iact)/theta - (1-theta)/theta * HI(Iact);

        % save head in head structure
        Phi(idt).time  = time(idt+1);            %#ok
        Phi(idt).values= reshape(HI,gr.size); %#ok

        % save cell by cell flows in budget structure
        for j=1:numel(cbcLbl)
            B(idt).time = t(idt)+dt(idt)*theta; %#ok
            [B(idt).label{j},B(idt).term{j}(Iact)] = updateB(B,cbcLbl{j}); %#ok

            B(idt).Psi(2:end,:) = cumsum(B(idt).term{j},1); %#ok
            B(idt).Psi(:,:)     = B(idt).Psi(end:-1:1,:);   %#ok
        end

    end
end

%% Service functions

%% compute conductances
function [Cx,Cy,Cs] = conductances(gr,Kx,Ky,Ss,theta)
    %CONDUCTANCE --- compute conductances
    % USAGE: [Cx,Cy] = contuctance(gr,Kx,Ky)
    
    if isscalar(Kx), Kx= gr.const(Kx); end
    if isscalar(Ky), Ky= gr.const(Ky); end
    if isscalar(Ss), Ss= gr.const(Ss); end
    
    % resistances and conducctances
    if ~gr.AXIAL
        RX = 0.5*bsxfun(@rdivide,gr.dx,gr.dy)./Kx;
        RY = 0.5*bsxfun(@rdivide,gr.dy,gr.dx)./Ky;
        Cx = 1./(RX(:,1:end-1)+RX(:,2:end));
        Cs = gr.Vol.*Ss/theta; Cs = Cs(:);
    else
        RX = bsxfun(@rdivide,log(gr.xGr(2:end-1)./gr.xm( 1:end-1)),2*pi*Kx(:,1:end-1).*gr.dY(:,1:end-1))+ ...
             bsxfun(@rdivide,log(gr.xm( 2:end  )./gr.xGr(2:end-1)),2*pi*Kx(:,2:end  ).*gr.dY(:,2:end  ));
        RY = 0.5*bsxfun(@rdivide,gr.dY./Ky, pi*(gr.xGr(2:end).^2 - gr.xGr(1:end-1).^2));
        Cs = gr.Vol.*Ss/theta; Cs = Cs(:);
        Cx = 1./RX;
    end
    Cy = 1./(RY(1:end-1,:)+RY(2:end,:));
end

%% parse input
function  [p,theta] = parser(varargin)
% parse input
    theta = 0.67;
    
    % optional parameters specified through param/value pair
    [theta,varargin] = getProp(varargin,'theta',theta);
    [p.RCH,  varargin] = getProp(varargin,'RCH',[]);
    [p.EVT,  varargin] = getProp(varargin,'EVT',[]);
    [p.GHB,  varargin] = getProp(varargin,'GHB',[]);
    [p.DRN,  varargin] = getProp(varargin,'DRN',[]);
    [p.RIV,  varargin] = getProp(varargin,'RIV',[]);
    [p.FQ ,  varargin] = getProp(varargin,{'FQ','WELLS'},[]);
    
    if ~isempty(varargin)
        warning('varargin not completely used'); %#ok
    end
end

%% set diagonal and RHS

function [D,RHS] = setD_RHS(stressLbl,D,RHS)
% setD_RHS -- set diagonal and RHS
iNr = 2; iQ=3; iH=3; iC=4; iHB=5;
global HT HI C Cs Iact dt idt
    switch stressLbl
        case 'STORAGE'
            RHS(Iact) = RHS(Iact) - Cs(Iact)/dt(idt).*HI(Iact); 
            D(Iact)   = D(Iact)   - Cs(Iact)/dt(idt);
        case 'FH'
            RHS(Iact) = RHS(Iact) - C(Iact,Ifh)*HI(Ifh);
        case 'GHB'
            I      = p.GHB{it}(:,iNr);
            RHS(I) = p.GHB{it}(:,iC).*p.GHB{it}(:,iH);
            D(  I) = p.GHB{it}(:,iC);           
        case 'DRN'
            I    = p.DRN{idt}(:,iNr);
            bdrn = HT(I)>p.DRN{idt}(:,iH);
            RHS(I) = RHS(I) + p.DRN(:,iC) .* bdrn .* p.DRN(:,iH);
            D(  I) = D(  I) + p.DRN(:,iC) .* bdrn;                       
        case 'RIV'
            I = p.RIV{idt}(:,iNr);
            briv =HT(I)>p.RIV(:,iHB);
            RHS(I) = RHS(I) + p.RIV(:,iC) .* (p.RIV(:,iH) - (~briv).* p.RIV(:,iHB));
            D(  I) = D(  I) + p.RIV(:,iC) .* briv;
        case 'WELLS'
            I = p.WELLS{idt}(:,iNr);
            RHS(I) = RHS(I) + p.WELLS{idt}(:,iQ);
        case 'RCH'
            RHS(Iact) = RHS(Iact) + p.RCH{idt}(Iact).*gr.Area(Iact);
        case 'EVT'
            RHS(Iact) = RHS(Iact) - p.EVT{idt}(Iact).*gr.Area(Iact);
        otherwise
            if ~ismember(stressLbl,...
                {'FLOWRIGHTFACE','FLOWFRONTFACE','FLOWLOWERFACE','STORAGE' 'CONSTANT HEAD'});
                error('illegal bcn label = %s\n',stressLbl);
            end
    end
end

%% update budget struct (compute specific cell by cell flows)

function Qcbc = updateB(stressLbl)
% getQcbc -- set stress boundary conditions into diagonal and RHS
global HT HI C diagC Cs
    iNr = 2; iQ=3; iH = 3; iC=4; iHB = 5;
    Qcbc = gr.const(0);
    switch stressLbl
        case 'NET'
            Qcbc = spdiags(diagC(~Iinact),0,C(~Iinact,~Iinact)) * HT(~Iinact);
        case 'FLOWRIGHTFACE'
            Qcbc =  -diff(HT,1,2) .* Cx;
        case 'FLOWFRONTFACE'
            Qcbc =  diff(HT,1,1).*Cy;
        case 'FLOWLOWERFACE'
            Qcbc = -diff(HT,1,3).*Cz;
        case 'STORAGE'        
            Qcbc = Cs(Iact)/dt(idt).*(HT(Iact)-HI(Iact));
        case 'FH'
            Qcbc  = C(Iact,Ifh)*HT(Ifh);
        case 'GHB'
            I = p.GHB{Idt}(:,iNr);
            Qcbc(I)  = p.GHB{idt}(:,iC) * HT(I) - p.GHB{idt}(:,iH);
        case 'DRN'
            I = p.DRN{idt}(:,iNr);
            bdrn    = HT(I)>p.DRN{idt}(:,iH);
            Qcbc(I) = p.DRN{idt}(:,iC) .* bdrn .* (p.DRN{idt}(:,iH) - HT(I));
        case 'RIV'
            I = p.RIV{idt}(:,iNr);
            briv    = HT(I)>p.RIV{idt}(:,iHB);
            Qcbc(I) = p.RIV{idt}(:,iC) .* (p.RIV{idt}(:,iH) - briv.*HT(I) - (~briv).* p.RIV{idt}(:,iHB));
        case 'WELLS'
            I = p.WELLS{idt}(:,iNr);
            Qcbc(I) = p.WELLS{idt}(:,iQ);
        case 'RCH'
            Qcbc = RCH(Iact).*gr.Area(Iact);
        case 'EVT'
            Qcbc = EVT(Iact).*gr.Area(Iact);
        otherwise
            error('illegal bcn label = %s\n',stressLbl);
    end
end

%% check parameter/value input pairs

function data = checkInput(lbl,data)
    switch lbl
        case 'FH'
           data = expand(data);
           if ~ismember(size(data,2),[3,5])
               error('input tuples for %s must be [iSP Lay Row Col H] or ] [iSP Idx H]');
           end
        case {'GHB','DRN'}
            data = expand(data);
            if ~ismember(size(data,2),[4,6])
               error('input tuples for %s must be [iSP Lay Row Col H C] or ] [iSP Idx H C]');
            end
        case 'RIV'
            data = expand(data);
            if ~ismember(size(data,2),[5,7])
               error('input tuples for %s must be [iSP Lay Row Col HRIV CRIV HRIVBOT] or ] [iSP Idx HRIV CRIV HRIVBOT]');
            end
        case 'RCH'
            if ~ismember(size(data,2),[5,7])
               error('input tuples for %s must be [iSP Lay Row Col HRIV CRIV HRIVBOT] or ] [iSP Idx HRIV CRIV HRIVBOT]');
            end
        case 'EVT'
            if ~ismember(size(data,2),[5,7])
               error('input tuples for %s must be [iSP Lay Row Col HRIV CRIV HRIVBOT] or ] [iSP Idx HRIV CRIV HRIVBOT]');
            end
        otherwise
    end        
end

%% expand param/value input into correct and complete format

function dataOut = expand(lbl,ndt,dataIn)
% expand dat into clean format
    if iscell(dataIn)
        dataIn = cell2list(dataIn);
        if size(dataIn) = gr.size
        end
        for idt=ndt:-1:1
            dataOut{idt} = dataIn(dataIn(:,1)==idt,:); % may be empty of no data in this period
            if isempty(dataOut{idt})
                continue
            end            
            if dataOut{idt}(end,end)<0   % same as in previous period?
                dataOut{idt} = dataOut{idt-1};
            end
            % use global index
            if size(dataOut{idt},2)>5
                dataOut{idt} = [dataOut{idt}(:,1) cellIndex(L,R,C,gr.size)  dataOut{idt}(:,4:end)];
            elseif strmp(lbl,'FH') && size(dataOut{idt},2)>4
                dataOut{idt} = [dataOut{idt}(:,1) cellIndex(L,R,C,gr.size) dataOut{idt}(:,end)];
            end               
        end
    end
end


