% This Matlab script generates all the variables of a correctly applied
% FIRE Minimization Scheme to a single LJ particle attracted to another
% (fixed) LJ particle.  The particle is released a distance "initx" from 
% the fixed LJ particle, and, when FIRE Energy minimization is applied,
% follows a specific trajectory towards the fixed particle that is
% completely dependent on the way the FIRE algorithm is being implemented.
% The other values tracked (velocity, dt, alpha, time to convergence) can
% also be used to look for correctness.  However, after a certain threshold
% of number of floating point calculations, small but growing deviations
% between the calculations is expected. 

initx = 2;
initv = 0;
init_dt = 0.005;
init_alpha = 0.1;
Nmin = 5;
finc = 1.1;
fdec = 0.5;
falpha = 0.99;
dt_max = 10*init_dt;
Etol = 1e-7;
Ftol = 1e-7;

global r_cutoff
global eps
global sig 
r_cutoff = 3.0;
eps = 1.0;
sig=1.0;
m=1;

v=initv;
x=initx;
dt = init_dt;
alpha = init_alpha;
fireon = 1;


Nsteps = 0; %Aaron apparently assumes this is zero! for initialization!
dE = 1000;
E = 1000;
F = 1000;

xval=[];
alphaval=[];
vval=[];
Energy=[];
dtval=[];
tcount = 1;

vval(tcount) = v;
xval(tcount)=x;
dtval(tcount) = dt;

while (abs(F)/3.0 > Ftol && dE > Etol && tcount < 400)
    Eold = E;
    
    % Velocity Verlet Integration
    [E,F] = ljpair(x);  
    v = v + F/(2*m)*dt;
    x = x + v*dt;
    [E,F] = ljpair(x);
    v = v + F/(2*m)*dt;
    
    %tracking values
    Energy(tcount)=E;    
    alphaval(tcount) = alpha;
    
    %FIRE
    
    if fireon

        P = F*v;
        v = (1-alpha)*v+alpha*sign(F)*abs(v);

        if P>0
            Nsteps = Nsteps + 1;
            if Nsteps > Nmin
                dt = min(dt*finc, dt_max);
                alpha = falpha*alpha;
            end
        else
            Nsteps = 0;
            dt = dt*fdec;
            v = 0;
            alpha = init_alpha;
        end;

 
    end;
    dE = abs(E-Eold); 
    Fcalc = abs(F)/3.0;
        
    %tracking values
    tcount = tcount +1;
    vval(tcount) = v;
    xval(tcount)=x;
    dtval(tcount) = dt;

    
  
end;

dlmwrite('matlab_x.out',xval, ',')



    
