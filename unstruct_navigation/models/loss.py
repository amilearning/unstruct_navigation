import torch
from torch import nn

import torch
import torch.nn as nn
import numpy as np


def nig_nll(gamma, v, alpha, beta, y):
    two_beta_lambda = 2 * beta * (1 + v)
    t1 = 0.5 * (torch.pi / v).log()
    t2 = alpha * two_beta_lambda.log()
    t3 = (alpha + 0.5) * (v * (y - gamma) ** 2 + two_beta_lambda).log()
    t4 = alpha.lgamma()
    t5 = (alpha + 0.5).lgamma()
    nll = t1 - t2 + t3 + t4 - t5
    return nll.mean()


# Normal Inverse Gamma regularization
# from https://arxiv.org/abs/1910.02600:
# > we formulate a novel evidence regularizer, L^R_i
# > scaled on the error of the i-th prediction
def nig_reg(gamma, v, alpha, _beta, y):
    reg = (y - gamma).abs() * (2 * v + alpha)
    return reg.mean()


def EvidentialRegression(dist_params, y, lamb=1.0):
    return nig_nll(*dist_params, y) + lamb * nig_reg(*dist_params, y)


# # def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
# #     twoBlambda = 2*beta*(1+v)
# #     nll = 0.5*torch.log(np.pi/v)  \
# #         - alpha*torch.log(twoBlambda)  \
# #         + (alpha+0.5) * torch.log(v*(y-gamma)**2 + twoBlambda)  \
# #         + torch.lgamma(alpha)  \
# #         - torch.lgamma(alpha+0.5)

# #     return torch.mean(nll) if reduce else nll

# # def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
# #     KL = 0.5*(a1-1)/b1 * (v2*torch.square(mu2-mu1))  \
# #         + 0.5*v2/v1  \
# #         - 0.5*torch.log(torch.abs(v2)/torch.abs(v1))  \
# #         - 0.5 + a2*torch.log(b1/b2)  \
# #         - (torch.lgamma(a1) - torch.lgamma(a2))  \
# #         + (a1 - a2)*torch.digamma(a1)  \
# #         - (b1 - b2)*a1/b1
# #     return KL

# # def NIG_Reg(y, gamma, v, alpha, beta, omega=0.01, reduce=True, kl=False):
# #     error = torch.abs(y-gamma)

# #     if kl:
# #         kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1+omega, beta)
# #         reg = error*kl
# #     else:
# #         evi = 2*v+(alpha)
# #         reg = error*evi

# #     return torch.mean(reg) if reduce else reg
    

# def EvidentialRegression(evidential_output, y_true , coeff=1.0):
    
#     #  mu, self.evidence(logv), self.evidence(logalpha) + 1, self.evidence(logbeta)
#     gamma =  evidential_output[:,0]
#     v =  evidential_output[:,1]
#     alpha =  evidential_output[:,2]
#     beta =  evidential_output[:,3]
#     loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta)
#     loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta)
#     return loss_nll + coeff * loss_reg



class EvidentialLossSumOfSquares(nn.Module):
  """The evidential loss function on a matrix.

  This class is implemented with slight modifications from the paper. The major
  change is in the regularizer parameter mentioned in the paper. The regularizer
  mentioned in the paper didnot give the required results, so we modified it 
  with the KL divergence regularizer from the paper. In orderto overcome the problem
  that KL divergence are missing near zero so we add the minimum values to alpha,
  beta and lambda and compare distance with NIG(alpha=1.0, beta=0.1, lambda=1.0)

  This class only allows for rank-4 inputs for the output `targets`, and expectes
  `inputs` be of the form [mu, alpha, beta, lambda] 

  alpha, beta and lambda needs to be positive values.
  """

  def __init__(self, debug=False, return_all=False):
    """Sets up loss function.
    Args:
      debug: When set to 'true' prints all the intermittent values
      return_all: When set to 'true' returns all loss values without taking average

    """
    super(EvidentialLossSumOfSquares, self).__init__()

    self.debug = debug
    self.return_all_values = return_all
    self.MAX_CLAMP_VALUE = 5.0   # Max you can go is 85 because exp(86) is nan  Now exp(5.0) is 143 which is max of a,b and l

  def kl_divergence_nig(self, mu1, mu2, alpha_1, beta_1, lambda_1):
    alpha_2 = torch.ones_like(mu1)*1.0
    beta_2 = torch.ones_like(mu1)*0.1
    lambda_2 = torch.ones_like(mu1)*1.0

    t1 = 0.5 * (alpha_1/beta_1) * ((mu1 - mu2)**2)  * lambda_2
    #t1 = 0.5 * (alpha_1/beta_1) * (torch.abs(mu1 - mu2))  * lambda_2
    t2 = 0.5*lambda_2/lambda_1
    t3 = alpha_2*torch.log(beta_1/beta_2)
    t4 = -torch.lgamma(alpha_1) + torch.lgamma(alpha_2)
    t5 = (alpha_1-alpha_2)*torch.digamma(alpha_1)
    t6 = -(beta_1 - beta_2)*(alpha_1/beta_1)
    return (t1+t2-0.5+t3+t4+t5+t6)

  def forward(self, inputs, targets):
    """ Implements the loss function 

    Args:
      inputs: The output of the neural network. inputs has 4 dimension 
        in the format [mu, alpha, beta, lambda]. Must be a tensor of
        floats
      targets: The expected output

    Returns:
      Based on the `return_all` it will return mean loss of batch or individual loss

    """
    assert torch.is_tensor(inputs)
    assert torch.is_tensor(targets)
    assert (inputs[:,1] > 0).all()
    assert (inputs[:,2] > 0).all()
    assert (inputs[:,3] > 0).all()

    targets = targets.view(-1)
    y = inputs[:,0].view(-1) #first column is mu,delta, predicted value
    a = inputs[:,1].view(-1) + 1.0 #alpha
    b = inputs[:,2].view(-1) + 0.1 #beta to avoid zero
    l = inputs[:,3].view(-1) + 1.0 #lamda
    
    if self.debug:
      print("a :", a)
      print("b :", b)
      print("l :", l)

    J1 = torch.lgamma(a - 0.5) 
    J2 = -torch.log(torch.tensor([4.0])) 
    J3 = -torch.lgamma(a) 
    J4 = -torch.log(l) 
    J5 = -0.5*torch.log(b) 
    J6 = torch.log(2*b*(1 + l) + (2*a - 1)*l*(y-targets)**2)
      
    if self.debug:
      print("lgama(a - 0.5) :", J1)
      print("log(4):", J2)
      print("lgama(a) :", J3)
      print("log(l) :", J4)
      print("log( ---- ) :", J6)

    J = J1 + J2 + J3 + J4 + J5 + J6
    #Kl_divergence = torch.abs(y - targets) * (2*a + l)/b ######## ?????
    #Kl_divergence = ((y - targets)**2) * (2*a + l)
    #Kl_divergence = torch.abs(y - targets) * (2*a + l)
    #Kl_divergence = 0.0
    #Kl_divergence = (torch.abs(y - targets) * (a-1) *  l)/b
    Kl_divergence = self.kl_divergence_nig(y, targets, a, b, l)
    
    if self.debug:
      print ("KL ",Kl_divergence.data.numpy())
    loss = torch.exp(J) + Kl_divergence

    if self.debug:
      print ("loss :", loss.mean())
    

    if self.return_all_values:
      ret_loss = loss
    else:
      ret_loss = loss.mean()
    #if torch.isnan(ret_loss):
    #  ret_loss.item() = self.prev_loss + 10
    #else:
    #  self.prev_loss = ret_loss.item()

    return ret_loss
  
  
class ContrastiveCorrelationLoss(nn.Module):
    """
    STEGO's correlation loss.
    """

    def __init__(
        self,
        cfg,
    ):
        super(ContrastiveCorrelationLoss, self).__init__()
        self.cfg = cfg

    def standard_scale(self, t):
        t1 = t - t.mean()
        t2 = t1 / t1.std()
        return t2

    def helper(self, f1, f2, c1, c2, shift):
        with torch.no_grad():
            # Comes straight from backbone which is currently frozen. this saves mem.
            fd = tensor_correlation(norm(f1), norm(f2))

            if self.cfg.pointwise:
                old_mean = fd.mean()
                fd -= fd.mean([3, 4], keepdim=True)
                fd = fd - fd.mean() + old_mean

        cd = tensor_correlation(norm(c1), norm(c2))

        if self.cfg.zero_clamp:
            min_val = 0.0
        else:
            min_val = -9999.0

        if self.cfg.stabilize:
            loss = -cd.clamp(min_val, 0.8) * (fd - shift)
        else:
            loss = -cd.clamp(min_val) * (fd - shift)

        return loss, cd

    def forward(
        self,
        orig_feats: torch.Tensor,
        orig_feats_pos: torch.Tensor,
        orig_code: torch.Tensor,
        orig_code_pos: torch.Tensor,
    ):
        coord_shape = [
            orig_feats.shape[0],
            self.cfg.feature_samples,
            self.cfg.feature_samples,
            2,
        ]
        coords1 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
        coords2 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1

        feats = sample(orig_feats, coords1)
        code = sample(orig_code, coords1)
        feats_pos = sample(orig_feats_pos, coords2)
        code_pos = sample(orig_code_pos, coords2)

        pos_intra_loss, pos_intra_cd = self.helper(feats, feats, code, code, self.cfg.pos_intra_shift)
        pos_inter_loss, pos_inter_cd = self.helper(feats, feats_pos, code, code_pos, self.cfg.pos_inter_shift)

        neg_losses = []
        neg_cds = []
        for i in range(self.cfg.neg_samples):
            perm_neg = super_perm(orig_feats.shape[0], orig_feats.device)
            feats_neg = sample(orig_feats[perm_neg], coords2)
            code_neg = sample(orig_code[perm_neg], coords2)
            neg_inter_loss, neg_inter_cd = self.helper(feats, feats_neg, code, code_neg, self.cfg.neg_inter_shift)
            neg_losses.append(neg_inter_loss)
            neg_cds.append(neg_inter_cd)
        neg_inter_loss = torch.cat(neg_losses, axis=0)
        neg_inter_cd = torch.cat(neg_cds, axis=0)

        return (
            pos_intra_loss.mean(),
            pos_intra_cd,
            pos_inter_loss.mean(),
            pos_inter_cd,
            neg_inter_loss,
            neg_inter_cd,
        )
