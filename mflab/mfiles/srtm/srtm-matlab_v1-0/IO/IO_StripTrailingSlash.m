function OutDirectory = IO_StripTrailingSlash(Directory)

% Remove any whitespace from the file name
OutDirectory = String_Trim(Directory);
    
% Remove the trailing '/' or '\' from the directory if needed
if (OutDirectory(length(OutDirectory)) == '/') || (OutDirectory(length(OutDirectory)) == '\')
    OutDirectory = OutDirectory(1: (length(OutDirectory) - 1) );
end