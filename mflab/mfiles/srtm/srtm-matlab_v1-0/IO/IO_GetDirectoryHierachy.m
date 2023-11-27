function hierarchy = IO_GetDirectoryHierachy(FullPath)
% Returns a cell array with the heirachy of directories. Will always
% contain at least one cell with the filename

[Dirname, filename, extension] = IO_BreakupFileName(FullPath);

% Split each of the paths in the directory
Remainder = Dirname;
hierarchy = cell(1);
i = 1;
while (~isempty(Remainder))
    [Token, Remainder] = strtok(Remainder, '\/');
    hierarchy{i} = Token;
    i = i + 1;
end

% The final part of the token is the filename itself
if (~isempty(extension))
    hierarchy{i} = [filename , '.', extension];
else
    hierarchy{i} = [filename];
end


    


