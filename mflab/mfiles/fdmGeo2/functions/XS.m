function A = XS(A)
%XS -- turns 3D array to cross section and vice versa

A = permute(A,[3,2,1]);