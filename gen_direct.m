%%**********************************************************************
% This function solves the semi-smooth Newton equation
%         (J + sigma * I) = -F(z).
% Input: 
%      model --- struct inherited from ssn
%     params --- struct inherited from ssn
%        out --- struct inherited from ssn
%
% Output: 
%          d --- the computed semi-smooth Newton direction
%      model --- struct inherited from ssn
%        out --- struct inherited from ssn
% ----------------------------------------------------------------------
% Author: Yiyang Liu, Zaiwen Wen
% Version 0.1 .... 2021/08
%%**********************************************************************
function [d, model, out] = gen_direct(model, params, out)
    H = model.sigma + 1 - double(model.D);
    invH = 1 ./ H;

    dtemp = invH .* -model.Fz;
    dtemp = (1 - 2 * double(model.D)) .* dtemp;
    dtemp = model.A_permu * dtemp;       
    v1 = model.sigma / (1 + model.sigma); v2 = 1 / v1;

    if isfield(model, 'D_old') && model.sigma_old == model.sigma && nnz(model.D ~= model.D_old) == 0
        % AKAT does not change at all
        if issparse(model.A) && model.n_densecols > 0 && model.n_denserows == 0
            dtemp2 = ldlsolve(model.LD_K1, dtemp);
            dtemp = model.A2K2sqrt_permu' * dtemp2;
            dtemp = model.LD_K2' \ (model.LD_K2 \ dtemp);
            dtemp = model.A2K2sqrt_permu * dtemp;
            dtemp = ldlsolve(model.LD_K1, dtemp);
            dtemp = dtemp2 - dtemp;
        elseif issparse(model.A)
            dtemp = ldlsolve(model.LD_K, dtemp); % LD is from LDL factorization
        else
            dtemp = model.LD_K' \ (model.LD_K \ dtemp); % LD is from LLT factorization
        end
    else
        % compute (A * K^(1 / 2))(chol_permu, :)
        AKsqrt_permu = sqrt(v1) * model.A_permu; 
        AKsqrt_permu(:, model.D == 1) = sqrt(v2) * model.A_permu(:, model.D == 1);
        if issparse(model.A) && model.n_densecols > 0 && model.n_denserows == 0
            model.A1K1sqrt_permu = AKsqrt_permu(:, model.idx_spcols);
            model.A2K2sqrt_permu = full(AKsqrt_permu(:, model.idx_densecols));
            
            if params.regAAT
                beta = 1e-12 * model.normA^2;
            else
                beta = 0 * model.normA^2;
            end
            
            [model.LD_K1] = ldlchol(model.A1K1sqrt_permu, beta);
            
            dtemp2 = ldlsolve(model.LD_K1, dtemp);
            dtemp = model.A2K2sqrt_permu' * dtemp2;
            
            % compute the dense n_densecols * n_densecols matrix
            M2 = model.A2K2sqrt_permu' * ldlsolve(model.LD_K1, model.A2K2sqrt_permu) + speye(model.n_densecols);
            [model.LD_K2, ~] = chol(M2, 'lower');
            dtemp = model.LD_K2' \ (model.LD_K2 \ dtemp);
            dtemp = model.A2K2sqrt_permu * dtemp;
            dtemp = ldlsolve(model.LD_K1, dtemp);
            dtemp = dtemp2 - dtemp;
        elseif issparse(model.A)
            p = nnz(model.D == 1);
            
            if params.regAAT
                beta = 1e-12 * model.normA^2;
            else
                beta = 0 * model.normA^2;
            end
            warning('off', 'all');
            if p < 0%size(model.A, 1) * 0.8
                model.LD_K = model.LD;
                model.LD_K(1 : size(model.LD_K, 1) + 1 : end) = model.LD(1 : size(model.LD, 1) + 1 : end) * v1;
                model.LD_K = ldlupdate(model.LD_K, sqrt(v2 - v1) * model.A_permu(:, model.D == 1));
            else
                model.LD_K = ldlchol(AKsqrt_permu, beta);
            end

            dtemp2 = ldlsolve(model.LD_K, dtemp);
            dtemp = dtemp2;
            warning('on', 'all');
        else
            AKAT_permu = AKsqrt_permu * AKsqrt_permu';
            if params.regAAT
                model.LD_K = chol(AKAT_permu + (1e-12 * model.normA^2) * speye(size(A, 1)), 'lower');
            else
                model.LD_K = chol(AKAT_permu, 'lower');
            end
            dtemp = model.LD_K' \ (model.LD_K \ dtemp);
        end
        out.nchol = out.nchol + 1;
    end 
    dtemp = model.A_permu' * dtemp;
    d = invH .* (dtemp - model.Fz);
end