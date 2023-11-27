function result = System_UnzipSingleFile(ZipFile, UnzipFile)
%
% Provides a system interface to unzip a file
if (isunix())
    
    % Unzip the file into a temporary directory
    TempDir = tempname;
    command = sprintf('unzip -j %s -d %s', ZipFile, TempDir);
    [status, result] = system(command);
    
    if (status ~= 0)
        error('Unzip operation failed %d: %s', status, result);
    end
    
    % Get the files from the directory
    DirectoryListing = dir(TempDir);
    
    % Check the number of files in that temp directory
    Files = length(DirectoryListing) - 2;
    if (Files ~= 1)
        warning('More than one file extracted, using the first file');
    end
    
    if (Files <= 0)
        error('No files unzipped');
    end
    
    % Move that file to the temp file
    TempFileName = strcat(TempDir, '/', DirectoryListing(3).name); 
    command = sprintf('mv %s %s', TempFileName, UnzipFile);
    system(command);
    
    % Delete the temporary files
    command = sprintf('rm -rf %s', TempDir);
    system(command);
    
    result = 1;
else
    error('Operating system not supported');
end


    
    
    

