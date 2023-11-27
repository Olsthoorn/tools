function o = makeShp(type,xy,hdrs,table)
%makeShp -- generates shape objects from data instead of form a shapefile
% USAGE: shpObj = makeShp(data]
%      type  = oneOf {'point','polyline','polygon'}
%      xy        = two column coordinate array or cell array of such arrays
%      hdrds = headers of attribute table of the shapes to be generated
%      table = table with attributes to extract values from. table(io,ia)
%      or      table{io,ia} where io is shapeObj nr and ia attribute numer
%
%      TO 140515

if isempty(xy)
    error('not enough input arguments');
end
if ~iscell(xy), xy={xy}; end

for io=numel(xy):-1:1    
    o(io)=shapeObj();
    o(io).Type     = strmatchi(type,{o(io).shapeTypes{:,2}}); 
    o(io).Type    = o(io).Type(1);
    o(io).TypeName = o(io).shapeTypes{o(io).Type,2};
    o(io).Points = xy{:};
    o(io).Box    = [min(o(io).Points(:,1)); ...
                    min(o(io).Points(:,2)); ...
                    max(o(io).Points(:,1)); ...
                    max(o(io).Points(:,2))];

    if exist('table','var')
        if iscell(table)
            for ih=1:numel(hdrs)
                o(io).UserData.(hdrs{ih}) = table{io,ih};
            end
        else
            for ih=1:nume(hdrs)
                o(io).UserData.(hdrs{ih}) = table(io,ih);
            end
        end
    end
end

 
 







