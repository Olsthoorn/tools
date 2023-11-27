function [I,Ix,Iy,i] = whichPoint(gr,I)
% Get indexes of MOC point in grid
% TO 140416
% I= gr.Nod*(N-1)+gr.Ny*(ix-1)+iy

if nargin<3, n=1; end

I = I-1; % Compute zero based, which is simpler than one-based:

i  = floor(I/gr.Nod);

I  = I - i*gr.Nod;

Ix = floor(I/gr.Ny);

Iy = I -  Ix*gr.Ny;

%export one-based
I = I+1;
i = i+1;
Ix= Ix+1;
Iy= Iy+1;

%gr.Nod * (i-1) + (Ix-1) * gr.Ny + Iy;