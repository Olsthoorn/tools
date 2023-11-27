function [value,varargin] = getProp(varargin,propertyName,default)
%GETPROP get property value from varargin, if not present use default
%
% USAGE: [value,varargin] = getProp(varargin,propertyName,default)
%
% if string propertyName is found in varargin{:}, the next value, i.e.
% varargin{i+1} is transferred to the output and varargin(i:i+1) are
% removed. If the propertyName does not exist in varargin, then varargin is
% untouched, while value becomes default.
%
% TO 130328

value = default;

if ~isempty(varargin)
%    i = find(strncmpi(propertyName,varargin,numel(propertyName)));
    i = strmatchi(propertyName,varargin);
    i = i(1);

    if i && numel(varargin)>i
        value = varargin{i+1};
        varargin(i:i+1) = [];
    end
end
