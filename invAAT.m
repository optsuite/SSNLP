%%**********************************************************************
% This function solves the linear system
%     (A * A') x = b
% using the predomputed Cholesky factorization of A * A'.
% Input:
%        b --- the rhs vector
%    model --- struct inherited from ssn
%
% Output:
%   result --- the computed x
% ----------------------------------------------------------------------
% Author: Yiyang Liu, Zaiwen Wen
% Version 0.1 .... 2021/08
%%**********************************************************************
function result = invAAT(b, model)
    % solve equation (A * A') * x = b;
    if issparse(model.A) && model.n_densecols > 0 && model.n_denserows == 0
        result2 = ldlsolve(model.LD1, b(model.permu));
        result = model.A2_permu' * result2;
        result = model.R2 \ (model.L2 \ result);
        result = model.A2_permu * result;
        result = ldlsolve(model.LD1, result);
        result = result2 - result;        
    elseif issparse(model.A)
%         result = model.R \ (model.L \ b(model.chol_permu));
        result = ldlsolve(model.LD, b(model.permu));
    else
        result = model.R \ (model.L \ b(model.permu));
    end
    result(model.permu) = result;
end