% Analysis of input and output of the chloride concentration
% of the Amsterdam Water Supply Dunes between 1980 and 2002
% The data are in the accompanying spreadsheet

defaults= {'nextPlot','add','xGrid','on','yGrid','on','fontSize',12};

xlsFiles = dir('*.xls');
FName = xlsFiles(strmatchi('Imp',{xlsFiles.name})).name;

[iHdr,iVals] = getExcelData(FName,'cl_in','Horizontal');
[oHdr,oVals] = getExcelData(FName,'cl_out','Horizontal');

I = strmatchi({'Date','cl_in'},iHdr);
iVals = iVals(:,I);
iHdr  = iHdr( :,I);

iVals(:,1) = excel2datenum(iVals(:,1));
oVals(:,1) = excel2datenum(oVals(:,1));

tStart = max(min(iVals(:,1)), min(oVals(:,1)));
tEnd   = min(max(iVals(:,1)), max(oVals(:,1)));

fprintf('Common datea are from %s to %s\n',datestr(tStart),datestr(tEnd));

In = iVals(iVals(:,1)>=tStart & iVals(:,1)<=tEnd,:);
Out= oVals(oVals(:,1)>=tStart & oVals(:,1)<=tEnd,:);

% check for need of interpolation
fprintf('Dt In  min= %10.1f max=%10.1f mean= %10.1f std=%10.1f\n',...
    min(diff(In( :,1))),max(diff(In( :,1))),...
    mean(diff(In( :,1))),std(diff(In( :,1))));
fprintf('Dt Out min= %10.1f max=%10.1f mean= %10.1f std=%10.1f\n',...
    min( diff(Out(:,1))),max(diff(Out(:,1))),...
    mean(diff(Out(:,1))),std(diff(Out(:,1))));

%% Show most common time step difference
% turns out to be 2 weeks
figure;
axes(defaults{:});
hist(diff(Out(:,1))); xlabel('Day separation'); ylabel('nr of times in data');
title('histogram of time between successive samples of chloride');

%%
figure('pos',screenPos(1));
ax = axes(defaults{:});
xlabel(ax,'time'); ylabel(ax,'chloride');
title(ax,'Chloride to and from Amsterdam Water Supply Dunes');
plot(ax,iVals(:,1),iVals(:,2),'b');
plot(ax,oVals(:,1),oVals(:,2),'r','linEwidth',2);
datetick()

%%
% Lets see which day was the 14-day samping day
day   = floor(now) - (0:13); % days before today

start = zeros(size(day)) + round(now);
last  = zeros(size(day)) + round(now);

fprintf('Occurrence of sequence of two weekly samples and number:\n');
for iDay=numel(day):-1:1
    dayHit     = rem(round(day(iDay))-Out(:,1),14)==0;
    hits(iDay) = sum(dayHit);
    s  = Out(find(dayHit,1,'first'),1);
    l  = Out(find(dayHit,1,'last'),1);
    if ~isempty(s), start(iDay)=s; end
    if ~isempty(l), last(iDay)=l ; end

    % Prints occurrecs of samples with 14 days ticks
    fprintf('%3s %4d %s %s\n',datestr(day(iDay),'ddd'),hits(iDay),datestr(start(iDay)),datestr(last(iDay)));
end

i = find(hits==max(hits));

t  = start(i):14:last(i);
fprintf('start and end times %s %s, %d hits\n',datestr(t(1)),datestr(t(end)),hits(i));

In = [t; interp1(iVals(:,1),iVals(:,2),t)]';
Out= [t; interp1(oVals(:,1),oVals(:,2),t)]';


% Now change the input values to given the mean of the concentratioins over the output samples
tIn = iVals(:,1);
for it = 1:numel(t)    
    In(i,2) = mean(iVals(tIn>t(i)-14 & tIn<=t(i),2));
end

%%
figure('pos',screenPos(1)); 
ax = axes(defaults{:});
xlabel(ax,'time'); ylabel(ax,'chloride [mg/L]');
title('chloride WRK (blue) and from dunes (red)');
plot(ax,t,In( :,2),'bo','markerSize',6,'markerFaceColor','b');
plot(ax,t,Out(:,2),'ro','markerSize',6,'markerFaceColor','r');
datetick(ax);

plot(ax,iVals(:,1),iVals(:,2),'b','lineWidth',0.25);

%% Using functions of the system identification toolbox
data = iddata(Out(:,2),In(:,2));

[V,V_sd] = getpvec(arx10101);

%% simulate the optimal model (which is arx(10,6,1) according to Aikaike)
data2  = iddata(Out(:,2)-mean(Out(:,2)),In(:,2)-mean(In(:,2)));
ys     = sim(arx1061,data2);

figure; hold on;
xlabel('time'); ylabel('chloride conc [mg/L]');
title('measured and simulated chloride Amsterdam Water Supply dunes');
plot(In( :,1),In(:,2),'bo');
plot(Out(:,1),Out(:,2),'ro');
plot(Out(:,1),ys.OutputData+mean(Out(:,2)),'g','lineWidth',2);
datetick();
grid on

%% ARX model, this can be done directly in the ident GUI
na = 10;
nb = 10;
nk = 1;
aik(na,nb) = NaN;
for i=na:-1:1
    for j=nb:-1:1
        m   = arx(data,'na',i,'nb',j,'nk',nk);
        aik(i,j) = aic(m);
    end
end
imin = find(aik(:)==min(aik(:)));
Nod = reshape(1:na*nb,[na nb]);
ib= ceil(imin/na);
ia= imin - (ib-1)*na;
fprintf('best: AIC(arx(na=%d,nb=%d,nk=%d)) = %f\n',ia,ib,nk,aik(imin))

m = arx(data,'na',ia,'nb',ib,'nk',nk);
figure;
axes('nextplot','add','xgrid','on','ygrid','on');
xlabel('two week samples'); ylabel('impulse response'); title('impulse response');
impulse(data);

%% Determine impulse response analytically using Hantush-bruggeman function
% IR = A/t exp(-a2/t-b2*t)

Dt = mean(diff(In(:,1)));
a  = 0.023; A = 0.87; a2 = 75; b2 = 0.003; d  = mean(Out(:,2));
p  = log([a A a2 b2 d]);
F  = @(p) Fun(p,In(:,1),In(:,2)-mean(In(:,2)),Out(:,2));

p  = lsqnonlin(F,p);
a  = exp(p(1)); A  = exp(p(2)); a2 = exp(p(3)); b2 = exp(p(4)); d  = exp(p(5));
fprintf('a = %g, A=%g a2=%g b2=%g d=%g\n',a,A,a2,b2,d);

[e,br,be,yHat]= Fun(p,In(:,1),In(:,2)-mean(In(:,2)),Out(:,2));

%
figure; axes(defaults{:}); set(gca,'xScale','lin');
xlabel('time'); ylabel('Chloride'); title('chloride Amsterdam Water Supply (General function)');
plot(In(:,1),In(:,2),'bs');
plot(Out(:,1),Out(:,2),'ro');
plot(Out(:,1),yHat,'c');
plot(Out(:,1),e,'g');
plot(In(:,1),filter(br,1,In(:,2)-mean(In(:,2)))+filter(be,1,e)+d,'m','linewidth',2);
datetick();
legend('In','Y','yHat','res','simulated');

%
figure;
subplot(2,1,1,defaults{:},'xScale','lin');
title('impulse response deterministic (General function)');
plot(Out(:,1)-Out(1,1),br,'b');  xlabel('days'); ylabel('response 14d block');
subplot(2,1,2,defaults{:},'xScale','lin');
title('impulse response error (General function)');
plot(Out(:,1)-Out(1,1),be,'r');  xlabel('days'); ylabel('response 14d block');

%% Determine impulse response analytically using Gamma function
% IR = A a^nt^(n-1)exp(-at)/Gamma(n)

Dt = mean(diff(In(:,1)));
a  = 0.023; A = 0.85; a2=0.0035; n=0.328; d  = mean(Out(:,2));
p  = log([a A a2 n d]);
F  = @(p) FunGamma(p,In(:,1),In(:,2)-mean(In(:,2)),Out(:,2));

p  = lsqnonlin(F,p);
a  = exp(p(1)); A  = exp(p(2)); a2=exp(p(3)); n = exp(p(4)); d  = exp(p(5));
fprintf('a = %g, A=%g a2=%g n=%g d=%g\n',a,A,a2,n,d);

[e,br,be,yHat]= FunGamma(p,In(:,1),In(:,2)-mean(In(:,2)),Out(:,2));

%
figure; axes(defaults{:}); set(gca,'xScale','lin');
xlabel('time'); ylabel('Chloride'); title('chloride Amsterdam Water Supply (Gamma function)');
plot(In(:,1),In(:,2),'bs');
plot(Out(:,1),Out(:,2),'ro');
plot(Out(:,1),yHat,'c');
plot(Out(:,1),e,'g');
plot(In(:,1),filter(br,1,In(:,2)-mean(In(:,2)))+filter(be,1,e)+d,'m','linewidth',2);
datetick();
legend('In','Y','yHat','res','simulated');

%
figure;
subplot(2,1,1,defaults{:},'xScale','lin');
title('impulse response deterministic (Gamma function)');
plot(Out(:,1)-Out(1,1),br,'b'); xlabel('days'); ylabel('response 14d block');
subplot(2,1,2,defaults{:},'xScale','lin');
title('impulse response error (Gamma function)');
plot(Out(:,1)-Out(1,1),be,'r'); xlabel('days'); ylabel('esponse 14d block');


