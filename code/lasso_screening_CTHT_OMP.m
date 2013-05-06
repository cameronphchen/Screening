function [rejection computation_time] = lasso_screening_CTHT_OMP(B,x,lambda,verbose,vt_feasible, oneSided)

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

% find the radius:
if isempty(vt_feasible)
    r = 1/lambda - 1/lm;
    if verbose
        fprintf(1,'    Initiating the test, sphere radius r=%1.2f\n', r);
    end
else
    r = norm(vt_feasible-q);
    if verbose
        fprintf(1,'  Initiating the test using the external feasible solution, sphere radius r=%1.2f\n', r);
    end
end

if r~=0
    used = false(size(B,2), 1);
    %tol = 1e-12;
    b = bm;
    used(ibm)=true;
    
    bTbi = B'*b;
    qd=q-(q'*b-1)*b-b;
    qdTbi=B'*qd;
    
    if oneSided
        [lb ib2]=max(qdTbi.*double(used==false));   % lb, inbm
        b2 = B(:,ib2);
        assert(b2'*qd>0);
    else
        [lb ib2]=max(abs(qdTbi).*double(used==false));   % lb, inbm
        b2 = B(:,ib2);
        b2 = b2*sign(b2'*qd);
    end
    %must explicitly exclude b from being reselected as b2, due to
    %numerical errors
    
    % used(ib2)=true;
    
    n1=b;
    n2=b2;
    phi1=(q'*n1-1)/r;
    phi2=(q'*n2-1)/r;
    t3=n1'*n2;
    
    n1Tbi=bTbi;
    n2Tbi=B'*n2;
    
    s1=(phi2*t3-phi1)/sqrt(1-phi2^2);
    s2=(phi1*t3-phi2)/sqrt(1-phi1^2);
    
    region2 = n2Tbi<phi2 & bTbi-t3*n2Tbi>=-sqrt(1-n2Tbi.^2)*s1;
    region3 = bTbi <phi1 & n2Tbi-t3*bTbi>=-sqrt(1- bTbi.^2)*s2;
    region4 = bTbi-t3*n2Tbi<-sqrt(1-n2Tbi.^2)*s1 & n2Tbi-t3*bTbi<-sqrt(1- bTbi.^2)*s2;
    
    region6 = n2Tbi>-phi2 & bTbi-t3*n2Tbi<=sqrt(1-n2Tbi.^2)*s1;
    region7 = bTbi >-phi1 & n2Tbi-t3*bTbi<=sqrt(1- bTbi.^2)*s2;
    region8 = bTbi-t3*n2Tbi>sqrt(1-n2Tbi.^2)*s1 & n2Tbi-t3*bTbi>sqrt(1- bTbi.^2)*s2;
    
    Ql = -(1-r)*ones(size(bTbi));
    Ql(region2) = -1+r*phi2*n2Tbi(region2)+r*sqrt(1-phi2^2)*sqrt(1-n2Tbi(region2).^2)+tol;
    Ql(region3) = -1+r*phi1*bTbi (region3)+r*sqrt(1-phi1^2)*sqrt(1-bTbi(region3).^2)+tol;
    Ql(region4) = -1+r/(1-t3^2)*((phi1-phi2*t3)*bTbi(region4)+(phi2-phi1*t3)*n2Tbi(region4))+r/(1-t3^2)*sqrt(1-t3^2+2*phi1*phi2*t3-phi1^2-phi2^2)*sqrt(1-t3^2+2*t3*bTbi(region4).*n2Tbi(region4)-bTbi(region4).^2-n2Tbi(region4).^2)+tol;
    
    Qu = (1-r)*ones(size(bTbi));
    Qu(region6)  =  1+r*phi2*n2Tbi(region6)-r*sqrt(1-phi2^2)*sqrt(1-n2Tbi(region6).^2)-tol;
    Qu(region7)  =  1+r*phi1*bTbi (region7)-r*sqrt(1-phi1^2)*sqrt(1-bTbi(region7).^2)-tol;
    Qu(region8)  =  1+r/(1-t3^2)*((phi1-phi2*t3)*bTbi(region8)+(phi2-phi1*t3)*n2Tbi(region8))-r/(1-t3^2)*sqrt(1-t3^2+2*phi1*phi2*t3-phi1^2-phi2^2)*sqrt(1-t3^2+2*t3*bTbi(region8).*n2Tbi(region8)-bTbi(region8).^2-n2Tbi(region8).^2)-tol;
    
    if oneSided
        rejection = (qTbi<Qu);
    else
        rejection =  ((qTbi<Qu)&(qTbi>Ql));
    end
    
else
    if oneSided
        rejection = qTbi<1-r;
    else
        rejection = abs(qTbi)<1-r;
    end
end
computation_time = toc(tic_start);

if verbose
    fprintf(1,'    The advanced dome test rejected %d codewords, comprising %1.2f%% of all codewords\n', sum(rejection==true), 100*sum(rejection==true)/size(B,2));
end



end
