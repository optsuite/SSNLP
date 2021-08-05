%%**********************************************************************
% This function adatively update t during ADMM or SSN
% Input: 
%      t --- current value of t
%      x --- primal variable
%      y --- dual variable
%      s --- redued cost
%    tol --- desired accuracy
%  model --- struct inherited from ssn
% params --- struct inherited from ssn
%
% Output:
%  t_new --- the new value of t
% ----------------------------------------------------------------------
% Author: Yiyang Liu, Zaiwen Wen
% Version 0.1 .... 2021/08
%%**********************************************************************
function t_new = update_t(t, x, y, s, iter, tol, model, params)
    geomean_pinf = geomean(model.pinf_history(iter - params.t_adjust_iter + 1 : iter));
    geomean_dinf = geomean(model.dinf_history(iter - params.t_adjust_iter + 1 : iter));
    pd_ratio = geomean_pinf / geomean_dinf;
    if geomean_dinf > tol * 0.5 && pd_ratio < 10
        t_new = min(t * params.t_rate, params.t_ub);
    elseif geomean_dinf > tol * 0.5 && pd_ratio < 100
        t_new = t;
    elseif geomean_dinf > tol * 0.5 && pd_ratio > 100
        t_new = max(t / params.t_rate, params.t_lb);
    elseif geomean_dinf < tol * 0.5 && geomean_pinf > tol
        t_new = max(t/params.t_rate, params.t_lb);
    elseif pd_ratio > 5
        t_new = max(t / params.t_rate, params.t_lb);
    elseif pd_ratio < 1 / 5
        t_new = min(t * params.t_rate, params.t_ub);
    else 
        t_new = t;
    end
    
    %%
%     delta_H = model.A' * (y - model.y_old);
%     delta_lambda = x - model.x_old;
%     delta_G = s - model.s_old;
%     alpha_SD = (delta_lambda' * delta_lambda) / (delta_H' * delta_lambda);
%     alpha_MG = (delta_H' * delta_lambda) / (delta_H' * delta_H);
%     if 2 * alpha_MG > alpha_SD
%         alpha = alpha_MG;
%     else
%         alpha = alpha_SD - alpha_MG / 2;
%     end
%     beta_SD = (delta_lambda' * delta_lambda) / (delta_G' * delta_lambda);
%     beta_MG = (delta_G' * delta_lambda) / (delta_G' * delta_G);
%     if 2 * beta_MG > beta_SD
%         beta = beta_MG;
%     else
%         beta = beta_SD - beta_MG / 2;
%     end
%     alpha_cor = (delta_H' * delta_lambda) / (norm(delta_H) * norm(delta_lambda));
%     beta_cor = (delta_G' * delta_lambda) / (norm(delta_G) * norm(delta_lambda));
%     eps_cor = 0.2;
%     if alpha_cor > eps_cor && beta_cor > eps_cor
%         t_new = sqrt(alpha * beta);
%     elseif alpha_cor > eps_cor
%         t_new = alpha;
%     elseif beta_cor > eps_cor
%         t_new = beta;
%     else
%         t_new = t;
%     end
    %%
%     if pd_ratio > 5
%         t_new = max(t / params.t_rate, params.t_lb);
%     elseif pd_ratio < 1 / 5
%         t_new = min(t * params.t_rate, params.t_ub);
%     else 
%         t_new = t;
%     end
end