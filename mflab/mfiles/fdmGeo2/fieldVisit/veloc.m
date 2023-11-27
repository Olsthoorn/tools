function Q = Qhsd(v,d,L,W)
%Qd -- compute dischrage over an overlet based on crest height and crest Length
% of the overlet.
% Notice that this must be done iteratively because we did not measure the static head
g = 9.81;
secPerDay = 86400;

vOld = v;
for i=1:200
    hs = d + v^2/(2*g);
    Cd = 0.85+(min(1.5,max(0.4,hs/L))-0.45)/(1.5-0.45)*(1.11-0.85) ;
    Cd = min(Cd,1);  % otherwise does not converge !
    v  = 1.7/d *Cd * (d+v^2/(2*g))^(3/2);
    dv = abs(v - vOld);
    fprintf('v = %g, dv = %g, Cd = %g\n',v,dv,Cd)
    if dv<0.001,
        Q = v * d * W * secPerDay;
        return;
    end
    vOld = v;
end

error('not converged !');