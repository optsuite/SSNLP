%%**********************************************************************
% This is the main functio of the SSN algorithm that solves the LP:
%    min c'*x, s.t. Ax = b, l <= x <= u,
% element of l is either -inf or 0, element of u is either positive of inf.
% Input: 
% A, b, c, l, u --- data from the LP instance
% params --- struct that stores parameters of the algorithms:
%               t_init: initial parameter t
%          lambda_init: initial lambda
%           t_adaptive: 0 or 1, whether or not t is fixed
%               regAAT: 0 or 1, whether or not to perturb A * A' when factorizing
%             admm_tol: stopping crit for ADMM
%                  tol: stopping crit for SSN
%           outputflag: 0 or 1, whether to print detailed info per iteration
%        t_adjust_iter: adjust t for every t_adjust_iter iterations
%   lambda_adjust_iter: adjust lambda for every lambda_adjust_iter iterations
%         admm_maxiter: maximal number of iterations of ADMM
%              maxiter: maximal number of iterations of SSN
% ssn_linesearch_ngrid: maximal number of linesearch trials per SSN iterations
%      admm_print_iter: print ADMM info for every admm_print_iter iterations
%           print_iter: print SSN info for every print_iter iterations
%
% Output: 
% out --- struct that stores info of the algorithm:
%          x: computed primal solution
%          y: computed dual solution
%          s: computed reduced cost
%          z: computed z
%  admm_iter: iterations of ADMM
%   ssn_iter: iterations of SSN
% total_iter: admm_iter + ssn_iter
% pobj, dobj: primal and dual objecties
%       pinf: primal infeasibility
%       dinf: dual infeasibility
%        gap: duality gap
%          t: value of parameter t from the last SSN iteration
%     lambda: value of parameter lambda from the last SSN iteration
% ----------------------------------------------------------------------
% Author: Yiyang Liu, Zaiwen Wen
% Version 0.1 .... 2021/08
%%**********************************************************************
function out = ssn(A, b, c, l, u, params)
    if nnz(l ~= 0 & l ~= -inf) > 0 || nnz(u < 0)
        error('ERROR: some element of l is not zero or -inf, or some element of u is negative.\n');
    end
    %% set up parameters
    params.t_rate = 1.2;
    params.t_lb = 1e-12;
    params.t_ub = 1e12;
    if ~isfield(params, 't_init') params.t_init = 5; end
    if ~isfield(params, 'lambda_init') lambda = 1e1; else lambda = params.lambda_init; end
    if ~isfield(params, 't_adaptive') params.t_adaptive = true; end
    if ~isfield(params, 'regAAT') params.regAAT = false; end
    if ~isfield(params, 'rescaleA') params.rescaleA = false; end
    if ~isfield(params, 'admm_tol') params.admm_tol = 1e-3; end
    if ~isfield(params, 'tol') params.tol = 1e-6; else params.tol = params.tol; end
    if ~isfield(params, 'outputflag') outputflag = 1; else outputflag = params.outputflag; end
    if ~isfield(params, 't_adjust_iter') params.t_adjust_iter = 10; end
    if ~isfield(params, 'lambda_adjust_iter') params.lambda_adjust_iter = 1; end
    if ~isfield(params, 'admm_maxiter') params.admm_maxiter = 10000; end
    if ~isfield(params, 'maxiter') params.maxiter = 10000; end
    if ~isfield(params, 'ssn_linesearch_ngrid') params.ssn_linesearch_ngrid = 5; end
    if outputflag
        if ~isfield(params, 'admm_print_iter') admm_print_iter = 1000; else admm_print_iter = params.admm_print_iter; end
        if ~isfield(params, 'print_iter') print_iter = 100; else print_iter = params.print_iter; end
    end
    %%
    t = params.t_init;
    A_origin = A;
    b_origin = b;
    c_origin = c;  
    l_origin = l; u_origin = u;
    normb_origin = norm(b_origin);
    
    if params.rescaleA scale_A = max(norm(A, 'fro'), 1); else scale_A = 1; end
    A = A / scale_A;
    model.A = A;
    model.normA = norm(A, 'fro');
    b = b / scale_A;
    if issparse(A)
        model.permu = analyze(A, 'row');
    else 
        model.permu = [1 : size(A, 1)]';
    end
    model.A_permu = A(model.permu, :);
    model.n_densecols = 0;
    model.n_denserows = 0;
    %% handle possible dense window of A * A'
    A_pattern = spones(A);    
    density_threshold = 0.1;
    model.idx_densecols = find(sum(A_pattern) > density_threshold * size(A, 1));
    model.n_densecols = nnz(model.idx_densecols);
    model.idx_denserows = find(sum(A_pattern, 2) > density_threshold * size(A, 2));
    model.n_denserows = nnz(model.idx_denserows);
    if issparse(A) && model.n_densecols > 0 && model.n_denserows == 0
        model.idx_spcols = find(sum(A_pattern) <= density_threshold * size(A, 1));
        model.A1_permu = model.A_permu(:, model.idx_spcols); % sparse part
        model.condA1 = sqrt(condest(model.A1_permu * model.A1_permu'));
        while model.condA1 > 1e3 && density_threshold < 1
            density_threshold = min(density_threshold * 1.2, 1);
            model.idx_spcols = find(sum(A_pattern) <= density_threshold * size(A, 1));
            model.A1_permu = model.A_permu(:, model.idx_spcols); % sparse part
%             model.condA1 = -1;
            model.condA1 = sqrt(condest(model.A1_permu * model.A1_permu'));
        end
        %%
%         [~, chol_flag] = ldlchol(model.A1_permu, 0);
%         chol_flag
%         sqrt(condest(model.A1_permu * model.A1_permu'))
%         while chol_flag > 0 && density_threshold < 1
%             density_threshold = min(density_threshold * 1.2, 1);
%             model.idx_spcols = find(sum(A_pattern) <= density_threshold * size(A, 1));
%             model.A1_permu = model.A_permu(:, model.idx_spcols); % sparse part
%             [~, chol_flag] = ldlchol(model.A1_permu, 0);
%             chol_flag
%         end
        %%
        model.idx_denserows = find(sum(A_pattern, 2) > density_threshold * size(A, 2));
        model.n_denserows = nnz(model.idx_denserows);
        model.idx_densecols = find(sum(A_pattern) > density_threshold * size(A, 1));
        model.n_densecols = nnz(model.idx_densecols);
        
        model.A2_permu = full(model.A_permu(:, model.idx_densecols)); % dense part
        model.normA1 = norm(model.A1_permu, 'fro');   
        if outputflag
            fprintf('n_densecols = %i, n_dense_rows = %i\n', model.n_densecols, model.n_denserows);
        end
    end
    clear A_pattern;
    %% do Cholesky factorization of A * A', based on the dense window situation
    if params.regAAT 
        beta = 1e-16 * model.normA^2;
    else
        beta = 0;
    end
    if issparse(model.A) && model.n_densecols > 0 && model.n_denserows == 0
        model.LD1 = ldlchol(model.A1_permu, 0);
        model.L2 = chol(model.A2_permu' * ldlsolve(model.LD1, model.A2_permu) + speye(model.n_densecols), 'lower');
        model.R2 = model.L2';
    elseif issparse(model.A)
        model.LD = ldlchol(model.A_permu, beta);
        LDdensity = nnz(model.LD) / numel(model.LD);
        if outputflag
            fprintf('nnz(LD) = %i, LD density = %.2e\n', nnz(model.LD), LDdensity);
        end
    else
        model.L = chol(model.A_permu * model.A_permu' + beta * speye(size(A, 1)), 'lower');
        model.R = model.L';
    end
    %%
    
    
    normb = norm(b);
    normc = norm(c);  
    scale_b = max(norm(b), 1);   
    scale_c = max(norm(c), 1);
%     scale_b = 1; scale_c = 1;
    b = b / (scale_b);
    c = c / scale_c;
    l = l / scale_b; u = u / scale_b;
   
%%
%     alpha = 1; % addtional parameter for ADMM, 0 < alpha < 2
    model.pinf_history = zeros(1, params.admm_maxiter);
    model.dinf_history = zeros(1, params.admm_maxiter);
    indices_free = (l == -inf); % indices where l is -infty
    indices_bottom = (l > -inf & u == inf);
    indices_box = (u < inf);
    u_box = u(indices_box);
    A_times_c = A * c;
    model.ATinvAATb = A' * invAAT(b, model);
    model.ATinvAATAc = A' * invAAT(A * c, model);
    admm_nprint = 0;
    
%     x_ADMM = model.ATinvAATb; x_ADMM = min(max(x_ADMM, l), u);
%     y_ADMM = invAAT(A * c, model); 
%     s_ADMM = c - A' * y_ADMM; 
%     s_ADMM(indices_free) = 0; s_ADMM(indices_bottom) = max(s_ADMM(indices_bottom), 0);
    
    %% initialize
    if isfield(params, 'x_init')
        x = params.x_init / scale_b;
    else
        x = zeros(size(A, 2), 1);
    end
    if isfield(params, 's_init')
        s = params.s_init / scale_c;
    else
        s = zeros(size(A, 2), 1);
    end
    if isfield(params, 'z_init')
        z_init = params.z_init / scale_b;
    end
    for admm_iter = 1 : params.admm_maxiter
        A_times_x_ADMM = A * x;
        y = invAAT(A_times_c - A * s + (b - A_times_x_ADMM)  / t, model);
        temp_ATy = A' * y;
        p_ADMM = c - temp_ATy - x / t;
        
        s(indices_bottom) = max(p_ADMM(indices_bottom), 0);
        p_ADMM_box = p_ADMM(indices_box);
        s_ADMM_box = p_ADMM_box - min(max(p_ADMM_box, -u_box / t), 0);
        s(indices_box) = s_ADMM_box;
        
        temp = temp_ATy + s - c;
        
        x = x + 1 * t * temp;
        pobj = c_origin' * (scale_b * x);
        dobj = scale_b * scale_c * (b' * y + u_box' * min(s_ADMM_box, 0));
        pinf = norm((scale_A * scale_b) * A_times_x_ADMM - b_origin) / (1 + normb_origin);       
        model.pinf_history(admm_iter) = pinf;
        dinf = scale_c * norm(temp) / (1 + normc);
        model.dinf_history(admm_iter) = dinf;
        gap = abs(pobj-dobj)/(1+abs(pobj)+abs(dobj));
        if outputflag && (admm_iter == 1 || mod(admm_iter, admm_print_iter) == 0 || ...
                max([gap, pinf, dinf]) < params.admm_tol || admm_iter == params.admm_maxiter)
            admm_nprint = admm_nprint + 1;
            if mod(admm_nprint, 20 * admm_print_iter) == 1
                admm_out_head = sprintf('\n%5s  %7s  %8s  %8s  %7s  %7s  %7s\n', 'iter', 't', 'pobj', 'dobj', 'gap', 'pinf', 'dinf');
                fprintf('%s', admm_out_head);
            end
            fm = '%.1e';
            admm_out_row = sprintf('%5s  %7s  %8s  %8s  %7s  %7s  %7s', num2str(admm_iter), ...
                num2str(t, fm), num2str(pobj, fm), num2str(dobj, fm), ...
                num2str(gap, fm), num2str(pinf, fm), num2str(dinf, fm));
            fprintf('%s\n', admm_out_row);
        end
        if max([gap, pinf, dinf]) < params.admm_tol
            break;
        end        
        if params.t_adaptive && mod(admm_iter, params.t_adjust_iter) == 0
            t = update_t(t, x, y, s, admm_iter, params.admm_tol, model, params);
        end
    end
    if params.admm_maxiter == 0
        admm_iter = 0;
    end
    
    if outputflag
        fprintf('\nSwitched to semi-smooth Newton method...\n');
    end
    
 %%
    if isfield(params, 'z_init')
        z = z_init;
    else
        z = x - t * s;
    end
    % compute x and Fz
    x = min(max(z, l), u);
    temp = A * (2 * x - z - t * c) - b;
    model.Fz = A' * (invAAT(temp, model)) + z - x + t * c;
%     
    eta1 = 0;
    eta2 = 0.9;
    
    gamma1 = 0.5;
    gamma2 = 0.8;
    gamma3 = 3;
    lam_adaptive = 1;
    Newt_iter_count = 0;
    
    model.pinf_history = zeros(1, params.maxiter);
    model.dinf_history = zeros(1, params.maxiter);
    out.nchol = 0;
    lambda_new = lambda;
    normFz_old = norm(model.Fz);
    model.normFz = norm(model.Fz);
    model.sigma = normFz_old * lambda;
    
    model.stepsize = 1;
    ssn_nprint = 0;
    %%
    for iter = 1 : params.maxiter
        model.D = (z > l) & (z < u);
        if iter > 1 && lambda_new == lambda && ...
                abs(model.normFz - normFz_old) / normFz_old < 1e-3 ...
                && norm(model.D_old - model.D) == 0  
            model.sigma = model.sigma_old;
        else
            normFz_old = model.normFz;
            model.sigma = model.normFz * lambda_new;
        end
        
        lambda = lambda_new;
        [d, model, out] = gen_direct(model, params, out);
        model.D_old = model.D;
        model.sigma_old = model.sigma;
        %% linesearch: first try stepsize from the last iteration
        [stepsize, z_new, x_new, invAATtemp, ratio, model] = linesearch(z, d, t, A, b, c, l, u, model, params);
        model.stepsize = stepsize;
        %%
        
        if (iter == 1) && (model.normFz_new/model.normFz)>1
            lambda_new = 10 * lambda;
        end
%         if ratio > eta1
        if model.normFz_new / model.normFz < 1000
            z = z_new;
            model.Fz = model.Fz_new;
            model.normFz = model.normFz_new;
            
            x = x_new;
            Newt_flag = 's';
        else
            Newt_flag = 'f';
        end   
        if Newt_flag == 's'
            Newt_iter_count = Newt_iter_count + 1;
        end
        %% compute dual variable y
        if Newt_flag == 's'
%             y = invAATmap(b/t-A * ((2*x_new-z_new)/t-c), model);
            y = -invAATtemp / t;
        else
            y = invAAT(b / t - A * ((2 * x - z) / t - c), model);
        end
        
        temp_ATy = A' * y;
        
        p = c - temp_ATy - x / t;      
        s(indices_bottom) = max(p(indices_bottom), 0);
        p_box = p(indices_box);
        s_box = p_box - min(max(p_box, -u_box / t), 0);
        s(indices_box) = s_box;
        %% small improvement on s
%         s(indices_bottom) = max(c(indices_bottom) - temp_ATy(indices_bottom), 0);
        %% another choice of s
%         s = (x - z) / t;
        %% compute optimality conditions
        pobj = scale_b * scale_c * (c' * x);
        dobj = scale_b * scale_c * (b' * y + u_box' * min(s(u < inf), 0));

        gap = abs(pobj - dobj) / (1 + abs(pobj) + abs(dobj));
        pinf = norm(scale_b * (A_origin * x) - b_origin) / (1 + normb_origin);
        dinf = scale_c * norm(c - temp_ATy - s) / (1 + normc);
        
        model.pinf_history(iter) = pinf;
        model.dinf_history(iter) = dinf; 
        if max([gap, pinf, dinf]) > params.tol && params.t_adaptive && mod(iter, params.t_adjust_iter) == 0
            t_new = update_t(t, x, y, s, iter, params.tol, model, params);
            w = x + t_new / t * (x - z) - t_new*c;
            tmp = w - A' * (invAAT(A * w - b, model));
            z = tmp + t_new / t * (z - x);       
            t = t_new;

            % compute x and Fz
            model.x_pre = x;
            x = min(max(z, l), u);
            temp = A * (2 * x - z - t * c) - b;
            model.Fz = A' * (invAAT(temp, model)) + z - x + t * c;
            model.normFz = norm(model.Fz);
        end   
        if outputflag && (iter == 1 || mod(iter, print_iter) == 0 || max([gap, pinf, dinf]) <= params.tol)
            ssn_nprint = ssn_nprint + 1;
            if mod(ssn_nprint, 20) == 1
                ssn_out_head = sprintf('\n%6s %7s %3s %7s %7s %8s %6s %8s %8s %7s %7s %7s %7s\n', ...
                    'iter', 't', 'flg', 'lambda', 'mu', 'ratio', 'stepsz', 'normF', 'pobj', 'dobj', 'gap', 'pinf', 'dinf');
                fprintf('%s', ssn_out_head);
            end
            
            fm = '%.1e';
            ssn_out_row = sprintf('%6s %7s %3s %7s %7s %8s %6s %8s %8s %7s %7s %7s %7s %7s\n', ...
                num2str(iter), num2str(t, fm), Newt_flag, ...
                num2str(lambda, fm), num2str(model.sigma, fm), num2str(ratio, fm), ...
                num2str(model.stepsize, '%.1f'), num2str(model.normFz, fm), ...
                num2str(pobj, fm), num2str(dobj, fm), ...
                num2str(gap, fm), num2str(pinf, fm), num2str(dinf, fm));
            fprintf('%s\n', ssn_out_row);
        end    
        if lam_adaptive && mod(iter, params.lambda_adjust_iter) == 0
            if ratio >= eta2
                lambda_new = max(gamma1 * lambda, 1e-12);
            elseif ratio >= eta1
                lambda_new = gamma2 * lambda;
            else
                lambda_new = min(gamma3 * lambda, 1e+12);
            end
        end
        if max([gap, pinf, dinf]) < params.tol
            fprintf('\nssn successfully reached within 1e%s tolerance: pobj = %.6e\n\n', ...
                num2str(log(params.tol) / log(10)), pobj);
            break;
        end
    end
    
    %% return result
    out.x = x * scale_b;
    out.y = y * (scale_c / scale_A);
    out.s = s * scale_c;
    out.Newt_iter_count = Newt_iter_count;
    out.admm_iter = admm_iter;
    out.ssn_iter = iter;
    out.total_iter = out.admm_iter + out.ssn_iter;
    out.pobj = pobj;
    out.dobj = dobj;
    out.gap = gap;
    out.pinf = pinf;
    out.dinf = dinf;
    out.t = t;
    out.lambda = lambda;
    out.z = z * scale_b;
    return;  
end
