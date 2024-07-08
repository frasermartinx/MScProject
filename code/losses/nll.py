import torch


#maybe incorporate prior function for variance?


def NLL(mean, log_var, targets):
    # Ensure positive variance using the softplus function
    var = torch.exp(log_var)
    nll_data = (0.5 * ((targets - mean) ** 2 / var)).sum()
    nll_var = (0.5 * torch.log(var)).sum()
    nll =  nll_data + nll_var 

    #prior
    prior = torch.distributions.Gamma(
        torch.tensor(1).to(mean.device),
        torch.tensor(1).to(mean.device)
    )
    log_prior = prior.log_prob(1./var).sum()
    total_nll = nll - log_prior
    return total_nll
 