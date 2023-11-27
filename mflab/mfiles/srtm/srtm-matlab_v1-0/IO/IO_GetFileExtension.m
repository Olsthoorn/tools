function [FileName, Extension] = IO_GetFileExtension(InputFileName)

% Break the string into tokens
[FileName, Extension] = strtok(InputFileName, '.');

% Check if the filename actually has an extension
if (strcmp(FileName,InputFileName) == 1)
    % There is no extension
    Extension = '';
    return;
end


% Repeat until the extension is null
while(length(Extension) ~= 0)
    % Tokenise the string
    [FileName, Extension] = strtok(Extension, '.');
end

% Assign the actual file name
FileName = InputFileName(1: length(InputFileName) - length(FileName) - 1);

% Assign the actual extension
Extension = InputFileName(length(FileName) + 1 : length(InputFileName));
    