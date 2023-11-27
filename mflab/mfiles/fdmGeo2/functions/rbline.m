function p = rbline(ax)
    %RBLINE -- draw a rubberband line and return the start and end points
    % USAGE: [p1,p2]=rbline([ax]);     uses current axes
    % Sandra Martinka 2002
    Nmax = 500; udata=[];

    if nargin<1, ax = gca; end

    set(ax,'nextPlot','add');

    
    p(Nmax,2) = 0;

    % Get first point
    i=1;
    set(gcf,'WindowButtonMotionFcn','','DoubleBuffer','on');

    k=waitforbuttonpress; if k>0, return; end

    udata.ax     = ax;
    pnt          = get(ax,'CurrentPoint');
    p(i,1:2)     = pnt(1,1:2);
    udata.p      = p(1:i,:);
    udata.h      = plot(p(1:i,1),p(1:i,2),'+:');
    set(gcf,'UserData',udata);
    
    set(gcf,'WindowButtonMotionFcn','','DoubleBuffer','on');
    
    % Loop until keypress
    for i=2:Nmax
        udata.p = p(1:i,:);
        set(gcf,'UserData',udata);
        k=waitforbuttonpress;
        if k>0  % keypress else mousepress
            p = p(1:i-1,:);
            return
        end

        pnt = get(ax,'CurrentPoint');
        p(i,1:2) = pnt(1,1:2);
        udata.p  = p(1:i,:);
        set(udata.h,'xData',udata.p(:,1),'yData',udata.p(:,2));      %plot starting point
    end
    
    p=p(1:i,:);
end

function wbmf
%WBMF -- window motion callback function
    udata = get(gcf,'UserData');
    pnt   = get(udata.ax,'CurrentPoint');
    udata.p(end,:) = pnt(1,1:2);
    set(udata.h,'XData',udata.p(:,1),'YData',udata.p(:,2));
end
