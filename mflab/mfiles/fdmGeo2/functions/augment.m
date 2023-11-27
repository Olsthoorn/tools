function B = augment(A)
%AUGMENT - add column and row for use in PCOLOR and SURF
%
% USAGE: B = augment(A)
%
% TO 140428

B = zeros(size(A,1)+1,size(A,2)+1);

B(1:end-1,1:end-1) = A;
