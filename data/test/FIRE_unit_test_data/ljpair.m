
function [E,F] = ljpair(x)
global r_cutoff;
global eps;
global sig;
if x < r_cutoff
    E = 4*eps*((sig/x)^12 - (sig/x)^6);
    F = 4*eps*(12*sig^12/x^13 - 6*sig^6/x^7);
else
    E = 0;
    F = 0;
end;    
 