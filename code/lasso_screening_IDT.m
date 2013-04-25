% run experiment for various lasso screening
% by Cameron P.H. Chen @Princeton 
% contact: pohsuan [at] princeton [dot] edu
% lasso_screening_IDT(training_data,testing_sample(:,1),0.4,1,[],0);
function [rejection computation_time] = lasso_screening_IDT(B,x,lambda,verbose,vt_feasible, oneSided)



% s is the number of iterations
s=2
tol=0;
computation_time = 0;

q = zeros(size(x,1),s);
r = zeros(s,1);
rejection = false(size(B,2),1);
used = false(size(B,2),1); 

tic_start = tic;

% find lambda max :lm and q1 
q(:,1) = x/lambda;
phi = nan(size(B,2),s);
phi(:,1)=B'*q(:,1);

if oneSided
    [lm ibm]=max(phi(:,1));   % lm, inbm
    bm = B(:,ibm);
    assert(bm'*x>0);
else
    [lm ibm]=max(abs(phi(:,1)));   % lm, inbm
    bm = B(:,ibm);
    bm = bm*sign(bm'*x);
end
lm = lm*lambda;

if verbose
    fprintf(1,'\n    Lasso problem parameters: lambda = %1.2f (lambda_max = %1.2f)\n', lambda, lm);
end

% find radius r(1)
if isempty(vt_feasible)
    r(1) = 1/lambda - 1/lm;
    if verbose
        fprintf(1,'    Initiating the test, sphere radius r=%1.2f\n', r);
    end
else
    r(1) = norm(vt_feasible-q(:,1));
    if verbose
        fprintf(1,'  Initiating the test using the external feasible solution, sphere radius r=%1.2f\n', r(1));
    end
end


for j1=1:s 
  [tmp h] = max(abs(phi(:,1)).*(rejection==false).*(used==false));
  b = sign(phi(h,1))*B(:,h);
  t = B'*b;
  psi = (abs(phi(h,1))-1)/r(j1);
  if psi <= 0
    break
  end
  if j1 < s
    q(:,j1+1) = q(:,j1) - psi*r(j1)*b;
    phi(:,j1+1) = phi(:,j1) - psi*r(j1)*t;
    r(j1+1) = r(j1)*sqrt(1-psi^2);
  end
  for j2 = j1:-1:1
    if j2 < j1
      psi = (q(:,j2)'*b -1)/r(j2);
    end 
    r_sc = r(j2);
    q_sc = q(:,j2);
    %%%%%
    Vl = -(1-r_sc)*ones(size(t));
    Vl(t <= (q_sc'*b-1)/r_sc) = -1+(q_sc'*b-1)*t(t <= (q_sc'*b-1)/r_sc) +...
                               sqrt(r_sc^2-(q_sc'*b-1)^2)*sqrt(1-t(t <= (q_sc'*b-1)/r_sc).^2)+tol;
    Vu = (1-r_sc)*ones(size(t));
    Vu(t >= -(q_sc'*b-1)/r_sc) = 1+(q_sc'*b-1)*t(t >= -(q_sc'*b-1)/r_sc) -...
                                sqrt(r_sc^2-(q_sc'*b-1)^2)*sqrt(1-t(t >= -(q_sc'*b-1)/r_sc).^2)-tol;
    
    if oneSided
      rejection = (phi(:,j2)<Vu) | rejection;
    else
      rejection = ((phi(:,j2)<Vu)&(phi(:,j2)>Vl)) | rejection;
    end
    %%%%
  end  
  used(h) = true;
end


computation_time = toc(tic_start);

if verbose 
    fprintf(1,'    The dome test rejected %d codewords\n', sum(rejection==true));
end
