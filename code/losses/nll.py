import torch


#maybe incorporate prior function for variance?


def NLL(mean, log_var, targets):
    # Ensure positive variance using the softplus function
    var = torch.exp(log_var)
    nll_data = 0.5 * ((targets - mean) ** 2 / var)
    nll_var = 0.5 * torch.log(var)
    nll = nll_var + nll_data
    return nll.mean()
 