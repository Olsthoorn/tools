% Drilling file, obtained from Pierre 8 May 2014
% This is the full database of drilling in the AWD, old and new.

F = '~/GRWMODELS/AWD/AWD-data/Boringen/boring.mat';
load(F)

% It is a struct with 5 fields, each field is a sctruct with fields, where
% each field is a 1d-numerical array or a 1d-cell array with the same 
% number of elements. In fact these subfields form tuples and are quite
% easy to access
% For instance the presence of the boorstaat of 24H717
%%
I = find(strncmpi('24H717',boring.boorgat.putcode,length('24H717')));
I = find(strncmpi('P057',boring.boorgat.putcode,length('24H717')));

flds = fieldnames(boring.boorgat);

for i=1:numel(flds)
    if iscell(boring.boorgat.(flds{i}))
        fprintf('%-20s',flds{i});
        fprintf(' %s',boring.boorgat.(flds{i}){I(:)});
        fprintf('\n');
    else
        fprintf('%-20s',flds{i});
        fprintf(' %g',boring.boorgat.(flds{i})(I(:)));
        fprintf('\n');
    end
end

getBoring('24H717.1','filter_bb',boring,'exact','recent');
getBoring('P57','filter_bb',boring,'exact','recent');
