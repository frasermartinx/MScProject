import torch


#maybe incorporate prior function for variance?


def NLL(mean, log_var, targets):
    # Ensure positive variance using the softplus function
    var = torch.exp(log_var) 
    nll_data = 0.5 * ((targets - mean) ** 2 / var)
    nll_var = (0.5 * log_var)
    nll =  (nll_data + nll_var)

    #prior
    #prior = torch.distributions.Gamma(
        #torch.tensor(10).to(mean.device),
        #torch.tensor(0.1).to(mean.device)
    #)
    #precision = 1./var
    ##precision = torch.clamp(precision, 1e-6, 1e6)
    #log_prior = prior.log_prob(1./var)
    total_nll = nll 
    return total_nll.mean()
 