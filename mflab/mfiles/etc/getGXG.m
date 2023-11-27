function [GLG,GVG,GHG,HY]=getGXG(h,t,varargin)
%GETGXG computes average lowest, spring and highest groundwater level
%
% USAGE:
%    [GLG,GVG,GHG,HY]=getGXG(h,t[,options][,plotmode])
% options is a serie of parmeter value pairs like
%       ...,'title',title,'xlabel',xlabel','ylabel',ylabel,...
% h=h(NSection,Nt)
%
% Computes average lowest (GLG), spring (GVG) and highest (GHG) ground water level
% from piezometer time series over the given time span.
%
% used in ..
%   mflab/examples/mf2005/DutchTop/HSV
%   mflab/examples/mf2005/DutchTop/NPARK
%
% the heads are presumed ordered in the direction of the time vector.
% Therefore, a large number of time series can be handled at once
% plot=0 or omitted altogether, don't plot
% plot=1 plot heads over time with points that were used to compute CxG
% plot=2 plot GxG for all sections with section number on x-axis
%
% De parameter HY stays for Hydrological Year.
%
% TO 101023

% Copyright 2009-2013 Theo Olsthoorn, TU-Delft and Waternet, without any warranty
% under free software foundation GNU license version 3 or later


[ttl, varargin] = getProp(varargin,'title','');
[ylbl,varargin] = getProp(varargin,'ylabel','');
[xlbl,varargin] = getProp(varargin,'xlabel','');

DV=datevec(t(:));  % [Y M D h min sec]
startYr = DV(1,1);
endYr   = DV(end,1) -1; % last hydrological year

aprilFirstStartYr = datenum(startYr, 4, 1);
endMarchEndYr     = datenum(endYr+1, 3, 31);

if t(  1)>=aprilFirstStartYr,
    startYr = startYr + 1;
end

if t(end)<endMarchEndYr
    endYr = endYr - 1;  % last complete hydrological year
end

%endYr = endYr - 1;
years=max(startYr,endYr-8):endYr;

%% Hydrological years 1-4 tot 31-3

April = 4;

for i = numel(years):-1:1
    year = years(i);
    HY(i).t1 = datenum(year    , April, 1);
    HY(i).t2 = datenum(year + 1, April, 1);
    HY(i).I  = find(t >= HY(i).t1 & t < HY(i).t2);
    
    % Store the heads measured on 14th or 28th each month
    
    % There may be an issue if in a real 14 day measurement series the
    % 14th and 28th dates shift a bit due to weekend days. So to be robust
    % some play must be allowed for in selecting the date. In our case, we
    % have probably always daily data (because they are simulated on a
    % daily basis) and this issue does not play. But this must be realized
    % in general. TO 150419
    
    % Store all measurements on 14th or 28ty during current hydrologic year
    HY(i).J = find(t >= HY(i).t1 & t < HY(i).t2 & (DV(:,3) == 14 | DV(:,3) == 28));
    
    % Spring head:

    % Theo originally took april first to determine the spring grw level.
    % HY(i).K=find(DV(:,1)==years(i) & DV(:,2)==4 & DV(:,3)==1); % april first in hydrological years
   
    % However, we now take the the mean of march 14, march 28 and april 14
    % see http://www.grondwaterdynamiek.wur.nl/NL/Parameters/
    % Store respective indices in vector K
    HY(i).K = find(DV(:, 1) == year & DV(:, 2) == 3 & DV(:, 3) == 14 | ...
                   DV(:, 1) == year & DV(:, 2) == 3 & DV(:, 3) == 28 | ...
                   DV(:, 1) == year & DV(:, 2) == 4 & DV(:, 3) == 14);
               
    % Sort the  heads on the 14th and 28th of current hydrological year
    % to allow taking the highest and lowest three of them
    % Also remember their locations [Is], to allow plotting them on their
    % correct time.
    [h_yr, Is] = sort(h(:, HY(i).J), 2);   
    HY(i).GLG  = h_yr(:, 1:3); % get the lowest three
    HY(i).GLGJ = Is(:, 1:3);   % get their location in vector J
    
    HY(i).GHG  = h_yr(:, end-2:end); % get the highest three
    HY(i).GHGJ = Is(:, end-2:end);   % and their location in vector J
    
    HY(i).GVG  = h(:, HY(i).K);  % Get the spring head values
    HY(i).GVGJ = HY(i).K;       % and the index in the overall series
end

GHG = mean([HY.GHG], 2);
GLG = mean([HY.GLG], 2);
GVG = mean([HY.GVG], 2);

%% Show data and points picked out for GXG computation

if ~isempty(varargin)
    fprintf('plotting')
    figure; hold on; xlabel(sprintf('%d - %d', years(1), years(end))); ylabel('head [m]'); grid on;
    ht = title('GLG(red), GVG(green) and GHG(blue)');
    
    if ~isempty(ylbl), ylabel(ylbl); end
    if ~isempty(xlbl), xlabel(xlbl); end
    if ~isempty(ttl)
        title([ttl ', ' get(ht,'string')]);
    end
    
    plot(t([1 end])',[GLG GLG],'r--');
    plot(t([1 end])',[GVG GVG],'g--');
    plot(t([1 end])',[GHG GHG],'b--');
    
    for i=1:length(HY)
        plot(t(HY(i).J),  h(HY(i).J), 'bo');
        plot(t(HY(i).J(HY(i).GHGJ)), HY(i).GHG, 'ro', 'markerFaceColor', 'r');
        plot(t(HY(i).K)            , HY(i).GVG, 'go', 'markerFaceColor', 'g');
        plot(t(HY(i).J(HY(i).GLGJ)), HY(i).GLG, 'bo', 'markerFaceColor', 'b');
    end
    plot(t, h, 'k');
    datetick('x', 'yyyy');

    legend('GLG', 'GVG', 'GHG', 'GLG', 'GVG', 'GHG', 'data');
end

%% Show GLG, GVG and GHG for all cross sections
