function getBoring(name,what,boring,varargin)
% getBoring --- gets info for drilling in AWD from file boring.mat
% USAGE getBoring(drillingName,what)
%   drillingName '24H717', 'C028' etc.
%   what any of fields of boring (type boring to see them)
% EXAMPLE
%   getBoring('24H717','boorgat',boring)
%    boring = struct boring loaded from boring.mat
%
% TO 140508

mainFlds = fieldnames(boring);

if ~strncmp(what,mainFlds,length(what))
    error('what must be one of these filenames: %s',sprintf(' %s',mainFlds{:})); %#ok
end

flds = fieldnames(boring.(what));
code = 'putcode';
if ~any(strncmpi(code,flds,length(code)))
    code = 'filtercode';
end

I = find(strncmpi(name,boring.(what).(code),length(name)));

if any(strncmpi('exact',varargin,length('exact')))
    I = I(cellfun(@length,boring.(what).(code)(I)) == length(name));
end

if any(strncmpi('recent',varargin,length('recent')))
    I = I(boring.(what).datum(I) == max(boring.(what).datum(I)));
end

if isempty(I)
    fprintf('Can''t find well/drilling ''%s'' for ''%s''\n',name,what);
    return;    
end

for i=1:numel(flds)
    if iscell(boring.(what).(flds{i}))
        fprintf('%-20s',flds{i});
        fprintf(' %s',boring.(what).(flds{i}){I(:)});
        fprintf('\n');
    else
        fprintf('%-20s',flds{i});
        fprintf(' %g',boring.(what).(flds{i})(I(:)));
        fprintf('\n');
    end
end
