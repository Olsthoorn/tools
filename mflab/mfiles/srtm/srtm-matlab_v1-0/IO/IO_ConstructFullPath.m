function filename = IO_ConstructFullPath(DirectoryList)
% Constructs a filename from a given heirachy of files

if (ispc())
    Delimiter = '\';
else
    Delimiter = '/';
end

filename = '';
for i = 1:(length(DirectoryList) - 1)
    filename = [filename, DirectoryList{i}, Delimiter];
end

filename = [filename, DirectoryList{length(DirectoryList)}];
