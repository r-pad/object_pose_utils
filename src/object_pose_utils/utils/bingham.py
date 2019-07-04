import torch
import numpy as np
from object_pose_utils.utils import to_np
from pybingham import bingham_F, bingham_dF

from itertools import permutations
perms = set(permutations(range(3)))

class BinghamConst(torch.autograd.Function):
    """
    Pytorch Bingham normalization constant function.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        #F = bingham_F(input.detach().numpy().astype(np.double))
        with torch.no_grad():
            F = bingham_F(to_np(input).astype(np.double))
        
        return torch.as_tensor(F, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        #Z = input.detatch().numpy()
        with torch.no_grad():
            Z = to_np(input)
            #dF = bingham_dF(Z)
            # Not sure if this always prevents the NANs? Need to check
            #z_i = np.argsort(Z)
            #dF = np.array(bingham_dF(Z[z_i]))
            #dF[z_i] = dF
            
            for p in perms:
                p = list(p)
                dF = np.array(bingham_dF(Z[p]))[np.argsort(p)]
                if(not np.any(np.isnan(dF))):
                    break

            if(np.any(np.isnan(dF))):
                print('BinghamConst: Gradient NaN')
                dF = np.zeros_like(dF)
        grad_input = grad_output.clone() * torch.as_tensor(dF, dtype=grad_output.dtype) 
        if(torch.cuda.is_available()):
            grad_input = grad_input.cuda()
        #grad_input *= torch.as_tensor(dF, dtype=grad_output.dtype)
        return grad_input 

def bingham_const(input): 
    return BinghamConst().apply(input)

def makeDuelMatrix(q):
    a, b, c, d = q 
    mat = torch.stack([torch.stack([a, -b, -c, -d]),
                       torch.stack([b,  a, -d,  c]), 
                       torch.stack([c,  d,  a, -b]),
                       torch.stack([d, -c,  b,  a])])
    return mat

def makeDuelMatrix2(q2):
    p, q, r, s = q2
    mat = torch.stack([torch.stack([p, -q, -r, -s]),
                       torch.stack([q,  p,  s, -r]), 
                       torch.stack([r, -s,  p,  q]),
                       torch.stack([s,  r, -q,  p])])
    return mat 

def makeBinghamM(q, q2 = None):
    q = q/q.norm()
    if q2 is None:
        return makeDuelMatrix(q)
    q2 = q2/q2.norm()
    return torch.mm(makeDuelMatrix(q),
                    makeDuelMatrix2(q2))

def bingham_likelihood(M, Z, label, return_exponent = False):
    Z = Z.clamp(max=0, min=-1000)
    eta = bingham_const(Z[1:]).float()
    if(torch.cuda.is_available()):
        eta = eta.cuda()

    Z = torch.diag(Z)
    MZMt = torch.bmm(torch.bmm(M, Z.repeat([1,1,1])), torch.transpose(M,2,1))
    if(torch.cuda.is_available()):
        MZMt = MZMt.cuda()
    bingham_p = torch.mul(label.transpose(1,0).unsqueeze(2),
    torch.matmul(label,MZMt.transpose(2,0))).sum([0])
    if(return_exponent):
        return bingham_p, eta 
    else:
        bingham_p = 1./eta*torch.exp(bingham_p)
    return bingham_p

def isobingham_likelihood(mean, sigma, label, return_exponent = False):
    M = makeBinghamM(mean).unsqueeze(0)
    zero = torch.zeros(1).float()
    if(torch.cuda.is_available()):
        zero = zero.cuda()
    Z = torch.cat([zero,-sigma, -sigma, -sigma])
    return bingham_likelihood(M, Z, label, return_exponent)

def duel_quat_bingham_likelihood(q1, q2, z, label, return_exponent = False):
    M = makeBinghamM(q1, q2).unsqueeze(0)
    zero = torch.zeros(1).float()
    if(torch.cuda.is_available()):
        zero = zero.cuda()
    Z = torch.cat([zero, z])
    return bingham_likelihood(M, Z, label, return_exponent)

def duel_loss_calculation(pred_q1, pred_q2, pred_z, true_r):
    lik_exp, eta = duel_quat_bingham_likelihood(pred_q1, pred_q2, pred_z, true_r, return_exponent = True)
    lik = 1./eta*torch.exp(lik_exp)
    loss = -(lik_exp - torch.log(eta))
    #if(lik != lik):
    #    raise ValueError('NAN lik: {} for mean {} and sigma {}'.format(lik, pred_mean, pred_sigma))
    return loss, lik 

def iso_loss_calculation(pred_mean, pred_sigma, true_r):
    lik_exp, eta = isobingham_likelihood(pred_mean, pred_sigma, true_r, return_exponent = True)
    lik = 1./eta*torch.exp(lik_exp)
    loss = -(lik_exp - torch.log(eta))
    #if(lik != lik):
    #    raise ValueError('NAN lik: {} for mean {} and sigma {}'.format(lik, pred_mean, pred_sigma))
    return loss, lik 

