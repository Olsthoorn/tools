function IO_BreakupDirectory(DirectoryName, Pieces, FileType, QuitAtEnd)

% Ensure that the number of pieces is an integer
Pieces = floor(Pieces);

% Supply some information to the user
disp(sprintf('Running BreakupDirectory: DirectoryName = %s, Pieces = %d', ...
    DirectoryName, Pieces));

% Perform error checking on the input
if( ischar(DirectoryName) == 0)
    disp('#error Directory name not a string');
    
    % Check if there are orders to quit
    if nargin == 3
        quit;
        return;
    else
        return;
    end
end

% Check for the need to append a slash to the source directory
if (DirectoryName(length(DirectoryName)) ~= '\') && ... 
   (DirectoryName(length(DirectoryName)) ~= '/') 
   
    % Concatenate the slash
    DirectoryName = strcat(DirectoryName, '/');
    
end

% Check if there is a filetype to append
if (nargin >= 3)
    
    % Check if the first character is a wildcard
    if(FileType(1) ~= '*')
        FileType = strcat('*', FileType);
    end
    
    % Append the file type to the directory
    DirectoryName = strcat(DirectoryName, FileType);
    
    % Ensure that the directory is started from one
    FirstFileIndex = 1;
    
else
    
    % Start from the 3rd Index
    FirstFileIndex = 3;
    
end
   

% Get specified directory
Directory = dir(DirectoryName)

% Check that the directory exists and
% Check the number of pieces
if Pieces < 1 || length(Directory) == 0
    
    % Display error indicies
    disp('# 0 0');
    
     % Check if there are orders to quit
    if nargin == 3
        quit;
        return;
    else
        return;
    end
    
end

% Check if only a single piece is applicable
if Pieces == 1
    disp(sprintf('# %d %d', FirstFileIndex, length(Directory)));
    
     % Check if there are orders to quit
    if nargin == 4
        quit;
        return;
    else
        return;
    end
    
end

% Process the first piece
disp(sprintf('# %d %d', FirstFileIndex, floor(length(Directory) / Pieces)));

% Check if there was only 2 pieces specified
if Pieces == 2
    
    % Display the position of the rest
    disp(sprintf('# %d %d', floor(length(Directory) / Pieces) + 1 , length(Directory)))
    
    % Check if there are orders to quit
    if nargin >= 3
        quit;
        return;
    else
        return;
    end
end

% Process almost all remaining pieces
for i = 2:(Pieces - 1)
    
    % Calculate the start and the end
    Start = (floor(length(Directory) / Pieces) * (i-1)) + 1; 
    End   = (floor(length(Directory) / Pieces) * (i));
    
    % Display the start and the end
    disp(sprintf('# %d %d', Start, End));
    
end

% Process the final Piece
Start = End + 1;
End = length(Directory);
disp(sprintf('# %d %d', Start, End));

% Quit, if ordered to do so.
if nargin >= 4
    quit;
    return;
else
    return;
end
    
    

