function [rejection computation_time] = lasso_screening_DT(B,x,lambda,verbose,vt_feasible, oneSided)

%Normalize x and B
%use outside guarantee for normalization, reduce time consumption
%dim=size(B,1);
%x = x./sqrt(sum(x.^2,1));
%B = B./(ones(dim,1)*sqrt(sum(B.^2,1)));
tol=0;
computation_time = 0;

tic_start = tic;

% Find lm and bm: 
q = x/lambda; 
qTbi=B'*x/lambda;  

if oneSided
    [lm ibm]=max(qTbi);   % lm, inbm
    bm = B(:,ibm);
    assert(bm'*x>0);
else
    [lm ibm]=max(abs(qTbi));   % lm, inbm
    bm = B(:,ibm);
    bm = bm*sign(bm'*x);
end
lm = lm*lambda;

if verbose 
    fprintf(1,'\n    Lasso problem parameters: lambda = %1.2f (lambda_max = %1.2f)\n', lambda, lm);
end

% Calculate radius: 
if isempty(vt_feasible)
    r = 1/lambda - 1/lm;
    if verbose
        fprintf(1,'    Initiating the sphere test, sphere radius r=%1.2f\n', r);
    end
else
    r = norm(vt_feasible-q);
    if verbose
        fprintf(1,'  Initiating the sphere test using the external feasible solution, sphere radius r=%1.2f\n', r);
    end
end

b = bm;
bTbi = B'*b;

Vl = -(1-r)*ones(size(bTbi));
Vl(bTbi <= (q'*b-1)/r) = -1+(q'*b-1)*bTbi(bTbi <= (q'*b-1)/r) + sqrt(r^2-(q'*b-1)^2)*sqrt(1-bTbi(bTbi <= (q'*b-1)/r).^2)+tol;
Vu = (1-r)*ones(size(bTbi));
Vu(bTbi >= -(q'*b-1)/r) = 1+(q'*b-1)*bTbi(bTbi >= -(q'*b-1)/r) - sqrt(r^2-(q'*b-1)^2)*sqrt(1-bTbi(bTbi >= -(q'*b-1)/r).^2)-tol;

if oneSided
    rejection = (qTbi<Vu);
else
    rejection = ((qTbi<Vu)&(qTbi>Vl));
end

computation_time = toc(tic_start);

if verbose 
    fprintf(1,'    The dome test rejected %d codewords\n', sum(rejection==true));
end
end
