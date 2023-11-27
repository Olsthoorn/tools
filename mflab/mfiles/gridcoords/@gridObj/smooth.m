function smooth(o,ax,jRow,Llay,Lcbd,LAYVAR,CBDVAR,varargin)
    % smooth(o,ax,jRow,Llay,Lcbd,LAYVAR,CBDVAR,varargin)
    %   plotting one section along the x-axis, through row jRow.
    %   used in plotXSec
    %
    %   ax   = axis handle
    %   jRow = row through which the cross section runs
    %   Llay logical vector indicating the layer to be plotted
    %   Icbd same for confining beds
    %   LAYVAR a 3D layer variable to be plotted
    %   CBDVAR a 3D confining bed variable to be plotted.
    %   varargin{:} options, see gridObj.
    %
    % SEE ALSO: gridObj.stair gridObj.plotXSec
    %
    % TO 130321
        
    grey = [0.8 0.8 0.8];
    
    zb = XS(o.ZBlay(jRow,:,:));
    zt = XS(o.ZTlay(jRow,:,:));

    % look for hlines in varargin
    i = strmatchi('hlines',varargin); i=i(1);
    
    % always plot top and bottom
    plot(ax,o.xc,zt(  1,:),'color',grey);
    plot(ax,o.xc,zb(end,:),'color',grey);

    if i
        hLinesColor = varargin{i+1};
        varargin(i:i+1)=[];
        
        if ischar(hLinesColor) && strcmpi(hLinesColor,'on')
                hLinesColor = grey;
        end
    
    
        if ~ischar(hLinesColor) || ~(strcmpi(hLinesColor,'off') || strcmpi(hLinesColor,'none'))
            for iLay = 1:o.Nlay
                plot(ax,o.xc,zb(iLay,:),'color',hLinesColor);
            end            
        end
    end
    
    i=strmatchi('vlines',varargin); i=i(1);
    if i
        vLinesColor = varargin{i+1};
        varargin(i:i+1)=[];
        
        if ischar(vLinesColor) && strcmpi(vLinesColor,'on')
                vLinesColor = grey;
        end
        
        if ~ischar(vLinesColor) ||  ~(strcmpi(vLinesColor,'off') || strcmpi(vLinesColor,'none'))
    
            for ix=1:o.Nx
                if ix==1
                    x = o.xc([1; 1]);
                    z = [zt(1,1); zb(end,1)];
                elseif ix==o.Nx
                    x = o.xc([end; end]);
                    z = [zt(1,end); zb(end,end)];
                else
                    x =  0.5*(o.xc(ix)+o.xc(ix+1));
                    z = [0.5*(zt(1,ix)+zt(1,ix+1)); ...
                         0.5*(zb(end,ix)+zb(end,ix+1))];
                end
                plot(ax,x,z,'color',vLinesColor);
            end
        end
    end
        
    if ischar(LAYVAR)
       for iLay=find(Llay(:)')
          color = LAYVAR(min(iLay,numel(LAYVAR))); 
          fill([o.xc o.xc(end:-1:1)],[zb(iLay,:) zt(iLay,end:-1:1)],color,'parent',ax,varargin{:});
       end
    else
        for iLay=find(Llay(:)')
            color = XS(LAYVAR(jRow,:,iLay));            
            fill([o.xc o.xc(end:-1:1)],[zb(iLay,:) zt(iLay,end:-1:1)],[color color(end:-1:1)],'parent',ax,varargin{:});
        end
    end
    
    if ~isempty(CBDVAR)
        zb = XS(o.ZBcbd(jRow,:,:));
        zt = XS(o.ZTcbd(jRow,:,:));

        if ischar(CBDVAR)        
           for iCbd=find(Lcbd(:)')
              color = CBDVAR(min(iCbd,numel(CBDVAR))); 
              fill([o.xc o.xc(end:-1:1)],[zb(iCbd,:) zt(iCbd,end:-1:1)],color,'parent',ax,varargin{:});
           end
        else
            for iCbd=find(Lcbd(:)')
                color = XS(CBDVAR(jRow,:,iCbd));
                fill([o.xc o.xc(end:-1:1)],[zb(iCbd,:) zt(iCbd,end:-1:1)],[color color(end:-1:1)],'parent',ax,varargin{:});
            end
        end
    end
