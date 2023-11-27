function stair(o,ax,j,Llay,Lcbd,LAYVAR,CBDVAR,varargin)
    % stair(o,ax,j,Llay,Lcbd,LAYVAR,CBDVAR,varargin)
    %   plotting one section along the x-axis, through row j.
    %   used in plotXSec
    %
    %   ax = axis handle
    %   j  = row through which the cross section runs
    %   Llay logical vector indicating the layer to be plotted
    %   Ldbd same for confining beds
    %   LAYVAR a 3D layer variable to be plotted
    %   CBDVAR a 3D confining bed variable to be plotted.
    %   varargin{:} options, see gridObj.
    %
    % SEE ALSO: gridObj.smooth gridObj.plotXSec
    %
    % TO 130321
    
    grey = [0.8 0.8 0.8];
    
    x           = NaN(1,2*o.Nx);
    x(1:2:end-1)= o.xGr(1:end-1);
    x(2:2:end)  = o.xGr(2:end);

    zb = NaN(o.Nlay,2*o.Nx);
    zb(:,1:2:end-1) = XS(o.ZBlay(j,:,:));
    zb(:,2:2:end  ) = XS(o.ZBlay(j,:,:));

    zt = NaN(o.Nlay,2*o.Nx);
    zt(:,1:2:end-1) = XS(o.ZTlay(j,:,:));
    zt(:,2:2:end  ) = XS(o.ZTlay(j,:,:)); 
    
    i = strmatchi('hlines',varargin);
    if i
        hcolor  = varargin{i+1};
        varargin(i:i+1)=[];
        if ischar(hcolor)
            if strcmpi(hcolor,'off') || strcmpi(hcolor,'none')
                plot(ax,x,zt(  1,:),'color',grey);
                plot(ax,x,zb(end,:),'color',grey);
            else
                if strcmp(hcolor,'on'), hcolor=grey; end
                plot(ax,x,zt(  1,:),'color',grey);
                for i=size(zb,1):-1:1
                    plot(ax,x,zb(i,:),'color',hcolor);
                end
            end
        else
            plot(ax,x,zt(  1,:),'color',grey);
            for i=size(zb,1):-1:1
                plot(ax,x,zb(i,:),'color',hcolor);
            end
        end
    else
        plot(ax,x,zt(  1,:),'color',grey);
        plot(ax,x,zb(end,:),'color',grey);
    end
    
    i = strmatchi('vlines',varargin);
    if i
        vcolor = varargin{i+1};
        varargin(i:i+1)=[];
        if ischar(vcolor)
            if strcmpi(vcolor,'on')
                vcolor=grey;
            end
            if ~(strcmpi(vcolor,'off') || strcmpi(vcolor,'none'))
                for ix=1:numel(x)
                    plot(ax,x([ix ix]),[zb(end,ix) zt(1,ix)],'color',vcolor);
                end
            end
        else
            for ix=1:numel(x)
                plot(ax,x([ix ix]),[zb(end,ix) zt(1,ix)],'color',vcolor);
            end
        end
    end
                        
    x2 = [x x(end:-1:1)];
    
    if any(Llay)
        if ischar(LAYVAR)
            for iLay=find(Llay(:)')
                z2 = [zb(iLay,:) zt(iLay,end:-1:1)];
                color = LAYVAR(min(iLay,numel(LAYVAR)));
                fill(x2,z2,color(1),'parent',ax,varargin{:});
            end
        else
            C = NaN(o.Nlay,numel(x));
            C(:,1:2:end-1) = XS(LAYVAR(j,:,:));
            C(:,2:2:end  ) = XS(LAYVAR(j,:,:));
            for iLay=find(Llay(:)')
                z2 = [zb(iLay,:) zt(iLay,end:-1:1)];
                fill(x2,z2,[C(iLay,:) C(iLay,end:-1:1)],'parent',ax,varargin{:});
            end
        end
    end    
    
    if any(Lcbd)
        
        zt = NaN(o.Ncbd,numel(x));
        zb = NaN(o.Ncbd,numel(x));
        
        zt(:,1:2:end-1) = XS(o.ZTcbd(j,:,:));
        zt(:,2:2:end  ) = XS(o.ZTcbd(j,:,:));
        zb(:,1:2:end-1) = XS(o.ZBcbd(j,:,:));
        zb(:,2:2:end  ) = XS(o.ZBcbd(j,:,:));
        
        if ischar(CBDVAR)
            for iCbd=find(Lcbd(:)')
                color = CBDVAR(min(iCbd,numel(CBDVAR)));
                fill(x2,[zb(iCbd,:) zt(iCbd,end:-1:1)],color,'parent',ax,varargin{:});
            end
        else
            C = NaN(o.Ncbd,numel(x));
            C(:,1:2:end-1) = XS(CBDVAR(j,:,:));
            C(:,2:2:end  ) = XS(CBDVAR(j,:,:));
            for iCbd=find(Lcbd(:)')
                fill(x2,[zb(iCbd,:) zt(iCbd,end:-1:1)],[C(iCbd,:) C(iCbd,end:-1:1)],'parent',ax,varargin{:});
            end
        end
    end
end

