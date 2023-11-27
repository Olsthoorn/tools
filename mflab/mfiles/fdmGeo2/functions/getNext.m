function [value,varargin] = getNext(varargin,type,default)
%GETPROP get property value from varargin if it has given type, if not present use default
%
% USAGE: [value,varargin] = getNext(varargin,type,default)
%
% if string propertyName is found in varargin{:}, the next value, i.e.
% varargin{i+1} is transferred to the output and varargin(i:i+1) are
% removed. If the propertyName does not exist in varargin, then varargin is
% untouched, while value becomes default.
%
% SEE ALSO: getNext, getProp, getType, getWord
%
% TO 130328

value = default;

if ~isempty(varargin)
    if strncmpi(type,class(varargin{1}),numel(type))
        value = varargin{1};
        varargin(1) = [];
    end
end
