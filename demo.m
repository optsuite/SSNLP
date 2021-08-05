%%**********************************************************************
% This is a demo program for calling ssn to solve a simple LP.
% ----------------------------------------------------------------------
% Author: Yiyang Liu, Zaiwen Wen
% Version 0.1 .... 2021/08
%%**********************************************************************
%% read data (AFIRO from the NETLIB dataset)
probname = 'AFIRO';
load(strcat(probname, '.mat'));
l = lbounds; u = ubounds;
l(l < -1e20) = -inf; u(u > 1e20) = inf;
%% set up stopping tolerance
tol = 1e-8;
fprintf('size of A is %ix%i, density of A = %.2e\n', size(A, 1), size(A, 2), ...
    nnz(A) / numel(A));

%% set up some SSN parameters
params = struct;
params.t_init = 5;
params.t_adaptive = 1;
params.regAAT = 0;
params.admm_tol = 1e-2;
params.tol = tol;
params.outputflag = 1;
params.t_adjust_iter = 10;
params.lambda_adjust_iter = 1;
params.admm_maxiter = 0;
params.maxiter = 10000;
params.print_iter = 10;
params.ssn_linesearch_ngrid = 5;
%% call SSN to solve
total_time = tic;
out = ssn(A, b, c, l, u, params);
total_time = toc(total_time);
%% check optimality
x = out.x;
y = out.y;
s = out.s;
pobj = c' * x;
dobj = b' * y + u(u < inf)' * min(s(u < inf), 0);
gap = abs(pobj - dobj) / (1 + abs(pobj) + abs(dobj));
pinf = norm(A * x - b) / (1 + norm(b));
if nnz(u < inf) > 0
    pinf = pinf + 1e100 * norm(min(x - l, 0)) + 1e100 * norm(max(x - u - 1e-12 * max(u(u < inf)), 0));
else
    pinf = pinf + 1e100 * norm(min(x - l, 0));
end
dinf = norm(A' * y + s - c) / (1 + norm(c)) + 1e100 * norm(s(l == -inf)) + 1e100 * norm(min(s(l == 0 & u == inf), 0));

fprintf('final result: pobj = %.6e, dobj = %.6e\n', pobj, dobj);
fprintf('total time: %.2e seconds.\n', total_time);
fprintf('admm_iter = %i, ssn_iter = %i, successful ssn_iter = %i\n', out.admm_iter, out.ssn_iter, out.Newt_iter_count);
fprintf('gap = %.0e, pinf = %.0e, dinf = %.0e\n', gap, pinf, dinf);
