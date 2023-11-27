function [success, message] = IO_RecursiveMkdir(DirectoryName)

% Under unix, this is really easy
if (isunix)
    command = sprintf('mkdir -p %s', DirectoryName);
    [success, message] = system(command);

% Is there an equivalent windows command?
elseif (ispc)
    % Wondering if it's easier?
    error('Windows not supported');
    
% Who knows what they're using!
else
    error('Operating system not supported')
end