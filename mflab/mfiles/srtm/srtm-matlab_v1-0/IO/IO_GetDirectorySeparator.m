function separator = IO_GetDirectorySeparator()

if (ispc)
    separator = '\';
elseif (isunix)
    separator = '/';
else
    error('Operating system not supported');
end
