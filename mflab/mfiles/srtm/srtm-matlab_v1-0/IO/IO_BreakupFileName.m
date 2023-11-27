function varargout = IO_BreakupFileName(InputFileName)
% IO_BreakupFileName
%
% Breaks up a file name into the file name, extension and (optionally) the
% directory
%
% Function Syntax:
% [FileName] = BreakupFileName(InputFileName);
% [FileName, Extension] = BreakupFileName(InputFileName);
% [Directory, Filename, Extension] = BreakupFileName(InputFileName);
%
% Inputs:
% InputFileName - Input file name to break up.  May be a file name, a
%   relative path or a full path
%
% Outputs:
% Directory - (Optional)  The path of the file given.  The trailing
%   slash/backslash will be removed.  If the path is not given as a part of
%   the file name, this will return an empty string.
% FileName - File Name part.  The path will be stripped, regardless of the
%   use of the Directory output argument.  If only a pathname is given as
%   the input argument, then this will return an empty string.
% Extension - Extension of the file name.  This will NOT include the
%   leading dot.  If the filename does not have an extension, then this
%   will return an empty string.
%
% Example calls (Blank listings are empty strings):
% -----------------------------------------------------------------------
% | Argument                  | Directory   | FileName      | Extension |
% -----------------------------------------------------------------------
% |  /directory/Filename.ext  | /directory  | Filename      | ext       |
% |  ./Directory/Filename.ext | ./Directory | Filename      | ext       |
% |  directory/FileName       | directory   | Filename      |           |
% |  ./Filename.Extension     | .           | Filename      | Extension |
% |  FileName                 |             | FileName      |           |
% |  FileName.ext             |             | FileName      | ext       |
% |  .HiddenFile              |             | .HiddenFile   |           |
% |  ~/.HiddenFile            | ~           | .HiddenFile   |           |
% |  FileName.Part.EXT        |             | Filename.Part | EXT       |
% | Filename.                 |             | Filename.      |           | 
% -----------------------------------------------------------------------

% Filename: 'IO_BreakupFileName.m'
% Author:   Damien Dusha
% Version:  1.1
% Last Modified: 1st November 2006
% Last Modified By: Damien Dusha
% Change History:   
% (v1.1 - 1/11/06, Damien Dusha)
%       - Changed name and incorporated into the IO_ Library
%
% (v1.0 - 17/5/06, Damien Dusha) 
%       - Initial Issue

% Check for validity of parameters
if (ischar(InputFileName) == 0)
    error('Input file name must be a string');
end

% Check for the output arguments
if (nargout > 3)
    error('Too many output arguments');
end


% Check if there is a directory character
i = length(InputFileName);
while (i > 0)
    
    % Look to see if there is a directory character
    if (InputFileName(i) == '/') || (InputFileName(i) == '\')
        break;
    end
    
    % Decrement the index
    i = i - 1;
end

% Check if a directory slash was found
if (i == 0)
    
    % Assign as an empty string
    Directory = '';
    
else
    
    % Assign the directory, except for the slash
    Directory = InputFileName( 1: (i-1) );
    
    % Check if there are any remaining characters
    if ( (i+1) == InputFileName)
        
        % The Input file was entirely a directory
        FileName = '';
        Extension = '';
 
        % Assign the outputs on the basis of the number of outputs
        switch nargout
            case 1
                varargout{1} = FileName;
            case 2
                varargout{1} = FileName;
                varargout{2} = Extension;
            case 3
                varargout{1} = Directory;
                varargout{2} = FileName;
                varargout{3} = Extension;
            otherwise
                varargout{1} = FileName;
        end
        
        % No more work to perform
        return;
        
    end
        
    % Remove the directory argument from the filename
    InputFileName = InputFileName( (i+1) : length(InputFileName) );
    
end


% What is left it the file name and, possibly, an extension
i = length(InputFileName);
while ( i > 0 )
    
    % Look for the dot
    if (InputFileName(i) == '.')
        break;
    end
    
    % Decrement the index
    i = i - 1;
end

% Perform work on the basis of the index of the period]
if (i == 0) || (i == 1)
    
    % Entirely the file name (either system file, or no extension)
    FileName = InputFileName;
    Extension = '';
    
% Contains an extension
else
    
    % Assign the file name
    FileName = InputFileName(1: (i-1) );
    
    % Check that the last character is not a dot
    if (i == length(InputFileName) )
        FileName = strcat(FileName, '.');
        Extension = '';
    else
        Extension = InputFileName( (i+1) : length(InputFileName) );
    end
    
end

% Assign the outputs
switch nargout
    case 1
        varargout{1} = FileName;
    case 2
        varargout{1} = FileName;
        varargout{2} = Extension;
    case 3
        varargout{1} = Directory;
        varargout{2} = FileName;
        varargout{3} = Extension;
    otherwise
        varargout{1} = FileName;
end

        
       
    