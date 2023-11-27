classdef filterObj
    % filterObj as field of boringObj for boring AWD from boring.mat
    % the AWD boring database of Pierre Kamps
    % TO 140518
    properties
        name
        nr
        bb
        ok
        bk
        diamF, diamB
        wvp, isocode
        material
        UserData
    end
    properties (Dependent=true)
        heart
    end
    methods
        function o = filterObj(filterName,boring)
            if nargin==0, return; end
            %% get most recent bb
            I = find(strncmpi(filterName,boring.filter_bb.filtercode,length(filterName)));
            I = I(cellfun(@length  ,boring.filter_bb.filtercode(I))==length(filterName));
            
            I = I(boring.filter_bb.datum(I) == max(boring.filter_bb.datum(I)));
            
            o.name = filterName;
            
            o.bb = boring.filter_bb.filter_bb(I);
            
            % get bk and ok of filter
            I         = find(strncmpi(filterName,boring.filter.filtercode,length(filterName)));
            I         = I(cellfun(@length,boring.filter.filtercode(I))==length(filterName));
            o.nr      = boring.filter.filternummer(I);
            o.bk      = boring.filter.filter_bk(I);
            o.ok      = boring.filter.filter_ok(I);
            o.diamF   = boring.filter.filter_diam{I};
            o.diamB   = boring.filter.diameter_buis(I);
            o.wvp     = boring.filter.wvp{I};
            o.isocode = boring.filter.isocode(I);
            o.material= boring.filter.filter_mat{I};
            
        end
        function plot(o,parent,s)
            plot(s([1 1]), [parent.mv o.bk],'b');
            plot(s([1 1]), [o.bk      o.ok],'r');
            plot(s(1),o.heart,'ro');
        end
        function plot3(o,parent,~)
            plot3(parent.x([1 1]), parent.y([1 1]), [parent.mv o.bk],'b');
            plot3(parent.x([1 1]), parent.y([1 1]), [o.bk      o.ok],'r');
        end
        function heart = get.heart(o), heart = 0.5 * (o.bk + o.ok); end
    end
end