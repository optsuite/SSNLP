%%**********************************************************************
% This function do a simple linesearch along the direction d
% Input:
%             z --- current iterate
%             d --- the semi-smooth Newton direction
%             t --- current penalty parameter of the augmented Lagrangian function
% A, b, c, l, u --- original data from the LP problem
%         model --- struct inherited from ssn
%        params --- struct inherited from ssn
%
% Output: 
%      stepsize --- the computed stepsize
%         z_new --- z + stepsize * d
%         x_new --- min(max(z_new, l), u)
%    invAATtemp --- can be used later to compute F(z_new)
%         ratio --- -(F(z_new)' * d) / norm(d)^2 / stepsize;
%         model --- struct inherited from ssn
% ----------------------------------------------------------------------
% Author: Yiyang Liu, Zaiwen Wen
% Version 0.1 .... 2021/08
%%**********************************************************************
function [stepsize, z_new, x_new, invAATtemp, ratio, model] = linesearch(z, d, t, A, b, c, l, u, model, params)
    stepsize = model.stepsize;
    ATinvAATAz = A' * invAAT(A * z, model);
    ATinvAATAd = A' * invAAT(A * d, model);
    w = t * (c - model.ATinvAATAc) - model.ATinvAATb;
    ATinvAATAw = -model.ATinvAATb;
    
    %% first try stepsize from last ssn iteration
    z_new = z + stepsize * d;
    x_new = min(max(z_new, l), u); 
    min_normF = norm(x_new - (z_new - ATinvAATAz - stepsize * ATinvAATAd) - (w - 2 * ATinvAATAw));
    %% then try other stepsizes
    for alpha = [params.ssn_linesearch_ngrid - 1 : -1 : 1] * (1 / params.ssn_linesearch_ngrid)
        ztemp = z + alpha * d;
        xtemp = min(max(ztemp, l), u);
        norm_Fztemp = norm(xtemp - (ztemp - ATinvAATAz - alpha * ATinvAATAd) - (w - 2 * ATinvAATAw));
        if norm_Fztemp < min_normF
            min_normF = norm_Fztemp;
            stepsize = alpha;
        elseif norm_Fztemp < model.normFz
            break;
        end
    end
    %%
    z_new = z + stepsize * d;
    x_new = min(max(z_new, l), u);
    temp = A * (2 * x_new - z_new - t * c) - b;
    invAATtemp = invAAT(temp, model);
    model.Fz_new = A' * invAATtemp + z_new - x_new + t * c;
    model.normFz_new = norm(model.Fz_new);
    %% compute the ratio
    ratio = -(model.Fz_new' * d) / norm(d)^2 / stepsize;
    % equivalent (but more efficient) way to compute the ratio
%     ratio = (d' * (x_new - z_new - w) - ATinvAATAd' * (2 * x_new - z_new)) / norm(d)^2 / stepsize;
end