classdef boringObj
    %BORINGOBJ -- drilling objects constructed from the AWD database
    % in boring.mat, received from Pierre Kamps on May 8, 2014
    % Convert boring database into a real database with key {putcode filter}
    %
    % boring = 
    %       boorgat: [1x1 struct] --> key is putcode
    %        filter: [1x1 struct] --> key is [putcode filternummer]
    %      maaiveld: [1x1 struct] --> key is putcode
    %     filter_bb: [1x1 struct] --> key is filtercode -> splitsen in [putnummer filter datum]
    %         geohm: [1x1 struct] --> key is [putcode kabelnr electrodenr]
    %
    % TO 140508
    properties        
        name
        Nfilt, Nmini, Nelec % Nr of filters, minifilters and electrodes
        x, y, mv, depth, type
        boreDate, diam, boreLog, boreSys, chloride
        filter = filterObj();
        UserData
    end
    properties (Dependent=true)
       % latlon
        L
    end
    methods
        function o = boringObj(name,boring)
            if nargin==0, return; end
            
            if   ischar(name), name = {name}; end

            N = numel(name);
            
            for io=N:-1:1
                %% Get the borehole info
                o(io).name    = name{io};
                I             = find(strncmpi(name{io},boring.boorgat.putcode,length(name{io})));
                if isempty(I), continue; end
                I = I(end);
                o(io).x           = boring.boorgat.x(I);
                o(io).y           = boring.boorgat.y(I);
                o(io).Nfilt       = boring.boorgat.filters(I);
                o(io).Nmini       = boring.boorgat.minifilters(I);
                o(io).Nelec       = boring.boorgat.electrodes(I);
                o(io).depth       = min(boring.boorgat.diepste_filt(I),boring.boorgat.boor_diepte(I));
                o(io).type        = boring.boorgat.type{I};
                o(io).boreDate    = boring.boorgat.boor_datum{I};
                o(io).diam        = boring.boorgat.boor_diam(I);
                o(io).boreLog     = strcmpi(boring.boorgat.boorgatmeting{I},'ja');
                o(io).boreSys     = boring.boorgat.boor_systeem{I};
                o(io).chloride    = boring.boorgat.chloride(I);

                I = find(strncmpi(name{io},boring.maaiveld.putcode,numel(name{io})));
                I = I(end);
                o(io).mv = boring.maaiveld.mv(I);

                o(io).filter(o(io).Nfilt) = filterObj(); 
                for iFilt=o(io).Nfilt:-1:1
                    filterName = [o(io).name '.' sprintf('%d',iFilt)];
                    o(io).filter(iFilt)=filterObj(filterName,boring);
                end
            end
        end
        function plot(o,xx,yy)
            %PLOT --- plots a boring (drilling)
            % USAGE bore.plot()       % distance from bore 1 along profile
            %       bore.plot(xx,yy)  % distance to line xx,yy (e.g. coast)
            %  xx = vector of 2 coordinates of line
            %  yy = vector of 2 coordinates of line
            %
            % TO 140508
            
            if nargin<2
                s = cumsum([0 sqrt(diff([o.x]).^2+diff([o.y]).^2)]);
            else
                s = o.dist(xx,yy);
            end
            
            set(gca,'nextPlot','add','xLim',[-100 s(end)+100]);
            
            for io=1:numel(o)
                for iFilt = 1:numel(o(io).filter)
                    o(io).filter(iFilt).plot(o(io),s(io));
                end
                text(s(io),o(io).depth-3,o(io).name,'rotation',90,...
                    'HorizontalAlignment','right','VerticalAlignment','middle');
            end
        end
        function L = get.L(o)
            %L computes length of piezometer
            L = o.mv - min([o(1).filter.ok]);
        end
        function display(o)
            for io=1:numel(o)
                fldNmsA = fieldnames(o);
                for ifld = 1:numel(fldNmsA)
                    fprintf(' %10s',fldNmsA{ifld});
                end
                fprintf('\n');

                for ifld=1:numel(fldNmsA)
                    if ischar(o(io).(fldNmsA{ifld}))
                        fprintf(' %-10s',o(io).(fldNmsA{ifld}));
                    elseif isnumeric(o(io).(fldNmsA{ifld}))
                        fprintf(' %10g',o(io).(fldNmsA{ifld}));
                    end
                end
                fprintf('\n\n');
                
                
                fldNmsB = fieldnames(filterObj());
                
                for ifld = 1:numel(fldNmsB)
                    fprintf(' %10s',fldNmsB{ifld});
                end
                fprintf('\n');
                for iFilt = 1:numel(o(io).filter)
                    for ifld = 1:numel(fldNmsB)
                        if ifld==2
                            fprintf(' %10g',o(io).x);
                            fprintf(' %10g',o(io).y);
                            fprintf(' %10g',o(io).mv);
                        end
                        if ischar(o(io).filter(iFilt).(fldNmsB{ifld}))
                            fprintf(' %-10s',o(io).filter(iFilt).(fldNmsB{ifld}));
                        elseif isnumeric(o(io).filter(iFilt).(fldNmsB{ifld}))
                            fprintf(' %10g',o(io).filter(iFilt).(fldNmsB{ifld}));
                        end
                    end
                    fprintf('\n');
                end
                fprintf('\n');
            end
        end
   %     function latlon=get.latlon(o), [lon lat] = wgs2rd(o.x,o.y); latlon=[lat lon]; end
        function [r,x,y] = dist(o,xx,yy)
            %DIST -- distance from this borehole to line xx,yy
            % USAGE: r = o.dist(xx,yy)
            %  where xx = vector of two x-coordinates
            %  where yy = vector of two y-coordinates
            %
            % TO 140509
            
            e = [diff(xx); diff(yy)] / sqrt(diff(xx).^2+diff(yy).^2);
            U = [-e(2) -e(1);
                  e(1) -e(2)];
              
            for io=numel(o):-1:1
                lam = U \ [xx(1) - o(io).x;
                           yy(1) - o(io).y];
                x = o(io).x   - e(2) * lam(1); % intersection point
                y = o(io).y   + e(1) * lam(1); % intersection point
    %           x1 = xx(1) + e(1) * lam(2);
    %           y1 = yy(1) + e(2) * lam(2);
                r(io) = lam(1); % distance
            end
        end
    end
end

