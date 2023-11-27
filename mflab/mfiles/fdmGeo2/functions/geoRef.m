function xy = geoRef(varargin)
    %geoRef -- reoreferences an image, real world digitizing images with 2 known points
    % USAGE: xy = geoRef(imageName,xy,plotOptions]
    %        xy = geoRef(A,[xE yE],'plotOptions)
    % Inputs
    %      imageName = name of image file
    %      xy        = [x1 y1; x2 y2] known world coordinates of two points on image
    %      xE,yE     = world coordinates of figure extnt (UL and LR)
    %                  second form loads image array (not filename) and allows
    %                  and uses extent of image previously determined
    %      plotOptions = options accepted by plot
    % Outputs
    %      xy        = [x y] of digitize points or
    %      xy        = cellArray each element of which has [x y] of succesively
    %                  digized objects
    % Any KEY to to finish current line, excludes ?the last point
    % RETURN  to to finish session
    %
    % TO 140515

    NL = 100;  % max number of lines that can be digitized (rather than while 1)

    % handle different input possibilities
    [fname,varargin] = getNext(varargin,'char','');
    if ~isempty(fname) % read from file and xy is world coordinates
                       % of 2 points to be digitized
        A = imread(fname);
        [xy,varargin] = getNext(varargin,'double',[]);
    else % assume A already presentand xy is world coords of image extent
        [A ,varargin] = getNext(varargin,'double',[]);
        [xy,varargin] = getNext(varargin,'double',[]);
    end
    
    % colors to be used for individually digitized lines
    clrs = 'brgkmcyw';

    % handle lineSpec
    plotOptions = varargin;
    if ~isempty(plotOptions)
        if isLineSpec(plotOptions{1})
            % first cell is a legal lineSpec
            lSpec = plotOptions{1}; plotOptions(1)=[];
            % find color in lineSpec
            ic    = ismember(plotOptions{1},clrs);
            % put color at first place in clrs (1st line is user color)
            clrs  = [clrs(c) clrs(~c)];
        else % defauls lineSpec
            lSpec ='bx--';
            ic    = 1;
        end
    else % default plotOptions and lineSpec
        plotOptions = {'markerFaceColor',[0.8 0.8 0.8]};
        lSpec = 'bx--';
        ic    = 1;
    end
    
    if ~all(size(xy)==[2 2])
        error('second argument must be a array of two vectors [xW(1) yW(1); xW(2) yW(2)]');
    end

    xW = xy(:,1);
    yW = xy(:,2);
    clear xy;
    xy{NL}=[]; % will obtain digitized lines

    figure('pos',screenPos(0.8));
    hold on;
     
    if isempty(fname) % digitize on already referenced image (xW and yW are extents)
        % show immage on its real world coordinate axes
        image(xW,yW,A);
        set(gca,'yDir','reverse');
        
        % digitize up to NL lines
        for iL=1:NL
            lSpec(ic) = mf_color(iL,clrs);
            [xy{iL},button] = getXY([lSpec,plotOptions]);
            if isempty(button), break; end
        end
    else % load image, reference points, then digitize on world coord axes
        A = imread(fname);
        image(A); set(gca,'yDir','reverse')

        % digitize at least two points as reference point coinciding with xW yW
        [xP, yP] = ginput(2);

        % conversion definition from pixel coords to real world coords
        x = @(xp)  xW(1) + (xp - xP(1))/diff(xP)*diff(xW);
        y = @(yp)  yW(1) + (yp - yP(1))/diff(yP)*diff(yW);

        % image extent in world coordinates
        xE = x([1 size(A,2)]);
        yE = y([1 size(A,1)]);

        % then reset image to its world coordinates
        hold off
        image(xE,yE,A); set(gca,'yDir','normal');
        hold on
        
        % digitize up to NL lines
        for iL=1:NL
            lSpec(ic)= mf_color(iL,clrs);
            [xy{iL},button]=getXY([lSpec,plotOptions]);
            if isempty(button), break; end
        end
    end 

    % if only one line was digitized, then do not use a cell array format
    % else leave as is
    xy = xy(1:iL);
    if numel(xy)==1
        xy=cell2list(xy);
    end
end

function [xy,b] = getXY(plotOptions)
    % get line while plotting it on screen
    % USAGE: [xy, button] = getXY(plotOptions)
    %      k = lineNumber
    %     xy = [x y];
    %  button= 1..3 amouse button, >3 keyNr
    % TO 140516
    Np = 1000; % maximum number of points that can be digitized
    x = NaN(Np,1); y=NaN(Np,1); % allocate
    j=0;
    while true
        j=j+1;
        if j==Np, break; end
        [xp, yp, b] = ginput(1); % digitize one point
        fprintf('%10.2f %10.2f %3d\n',xp,yp,b);
        if isempty(b)
            xy=[x(1:j-1) y(1:j-1)]; % if not finished add point
            break;
        else
            x(j) = xp; % store point
            y(j) = yp;
            if b==8 % backspace
                j=j-2;
                if j>0
                    delete(h)
                    h = plot(x(1:j),y(1:j),plotOptions{:});
                else
                    xy = [ [] [] ]; b=[];
                    break;
                end
            elseif b<=3 % any mouse click
                if j==1    % if first, then initiate plot of line
                    h = plot(x(j),y(j),plotOptions{:});
                else % just replace xData and yData of line (extends)
                    set(h,'xData',x(1:j),'yData',y(1:j));
                end
            else % any other keypress --> line is finished                
                    xy = [x(1:j-1) y(1:j-1)];
                    % get out and back to caller
                    break;
            end
        end
    end
end