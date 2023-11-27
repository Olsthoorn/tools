function OutFileName = IO_ChangeFileExtension(FileName, Extension)

% Get the extension
[FilePart, Ext] = strtok(FileName, '.');
while ( ~isempty(Ext))
    [FilePart, Ext] = strtok(Ext, '.');
end

% Move all but the extension to the output
OutFileName = FileName(1:(length(FileName) - length(FilePart)-1));

% Check if the first character of the extension is a dot
if (Extension(1) == '.')
    
    % Copy the extension to the file name
    OutFileName = strcat(OutFileName, Extension);
    
else
    
    % Copy the dot then the extension
    OutFileName = strcat(OutFileName, strcat('.',Extension));
end


