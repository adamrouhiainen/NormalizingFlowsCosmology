import utilities

import base64
import io
import time
import pickle
import math
import numpy as np
import pylab as pl
import quicklens as ql
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F

import packaging.version
if packaging.version.parse(torch.__version__) < packaging.version.parse('1.5.0'):
  raise RuntimeError('Torch versions lower than 1.5.0 not supported')


######################## Real NVP flow


def make_checker_mask(shape, parity,torch_device):
    checker = torch.ones(shape, dtype=torch.uint8) - parity
    checker[::2, ::2] = parity
    checker[1::2, 1::2] = parity
    return checker.to(torch_device)


class AffineCoupling(torch.nn.Module):
    def __init__(self, net, *, mask_shape, mask_parity, torch_device):
        super().__init__()
        self.mask = make_checker_mask(mask_shape, mask_parity,torch_device)
        self.net = net

    def forward(self, x):
        x_frozen = self.mask * x      
        x_active = (1 - self.mask) * x
        net_out = self.net(x_frozen.unsqueeze(1))
        s, t = net_out[:,0], net_out[:,1]
        fx = (1 - self.mask) * t + x_active * torch.exp(s) + x_frozen
        axes = range(1,len(s.size()))
        logJ = torch.sum((1 - self.mask) * s, dim=tuple(axes))
        return fx, logJ

    def reverse(self, fx):
        fx_frozen = self.mask * fx
        fx_active = (1 - self.mask) * fx  
        net_out = self.net(fx_frozen.unsqueeze(1))
        s, t = net_out[:,0], net_out[:,1]
        x = (fx_active - (1 - self.mask) * t) * torch.exp(-s) + fx_frozen
        axes = range(1,len(s.size()))
        logJ = torch.sum((1 - self.mask)*(-s), dim=tuple(axes))
        return x, logJ
    
    
def make_conv_net(*, hidden_sizes, kernel_size, in_channels, out_channels, use_final_tanh):
    sizes = [in_channels] + hidden_sizes + [out_channels]
    assert packaging.version.parse(torch.__version__) >= packaging.version.parse('1.5.0')
    assert kernel_size % 2 == 1, 'kernel size must be odd for PyTorch >= 1.5.0'
    padding_size = (kernel_size // 2)
    net = []
    for i in range(len(sizes) - 1):
        net.append(torch.nn.Conv2d(sizes[i], sizes[i+1], kernel_size, padding=padding_size,stride=1, padding_mode='circular'))
        #net.append(torch.nn.Conv2d(sizes[i], sizes[i+1], kernel_size, padding=padding_size,stride=1, padding_mode='zeros'))
        if i != len(sizes) - 2:
            net.append(torch.nn.LeakyReLU())
        else:
            if use_final_tanh:
                net.append(torch.nn.Tanh())
    return torch.nn.Sequential(*net)


def make_flow1_affine_layers(*, n_layers, lattice_shape, hidden_sizes, kernel_size, torch_device):
    layers = []
    for i in range(n_layers):
        parity = i % 2
        net = make_conv_net(
            in_channels=1, out_channels=2, hidden_sizes=hidden_sizes,
            kernel_size=kernel_size, use_final_tanh=True)
        coupling = AffineCoupling(net, mask_shape=lattice_shape, mask_parity=parity,torch_device=torch_device)
        layers.append(coupling)
    return torch.nn.ModuleList(layers)



######################## Priors


class SimpleNormal:
    def __init__(self, loc, var):
        self.dist = torch.distributions.normal.Normal(torch.flatten(loc), torch.flatten(var))
        self.shape = loc.shape
    def log_prob(self, x):
        logp = self.dist.log_prob(x.reshape(x.shape[0], -1))
        return torch.sum(logp, dim=1)
    def sample_n(self, batch_size):
        x = self.dist.sample((batch_size,))
        return x.reshape(batch_size, *self.shape)
    
    
    
class CorrelatedNormal:
    def __init__(self, loc, var,nx, dx,cl_theo,torch_device):
        self.torch_device=torch_device
        self.nx=nx
        self.dx=dx
        
        #normal distribution to draw random fourier modes
        self.dist = torch.distributions.normal.Normal(torch.flatten(loc), torch.flatten(var))
        self.rfourier_shape = loc.shape
        
        #create the array to multiply the fft with to get the desired power spectrum
        self.ells_flat = self.get_ell(self.nx, self.dx).flatten() 
        clfactor = np.interp( self.ells_flat, np.arange(0,len(cl_theo)), np.sqrt(cl_theo), right=0 ).reshape( self.rfourier_shape[0:2] )
        self.clfactor = torch.from_numpy(clfactor).float().to(torch_device)
        clinvfactor = np.copy(clfactor) 
        clinvfactor[clinvfactor==0] = 1. #TODO: should we to remove the monopole?
        clinvfactor = 1./clinvfactor
        self.clinvfactor = torch.from_numpy(clinvfactor).float().to(torch_device)
        
        #masks for rfft symmetries
        a_mask = np.ones((self.nx, int(self.nx/2+1)), dtype=bool)
        a_mask[int(self.nx/2+1):, 0] = False
        a_mask[int(self.nx/2+1):, int(nx/2)] = False
        b_mask = np.ones((self.nx, int(self.nx/2+1)), dtype=bool)    
        b_mask[0,0] = False
        b_mask[0,int(self.nx/2)] = False
        b_mask[int(self.nx/2),0] = False
        b_mask[int(self.nx/2),int(self.nx/2)] = False
        b_mask[int(self.nx/2+1):, 0] = False
        b_mask[int(self.nx/2+1):, int(self.nx/2)] = False
        self.a_mask = a_mask
        self.b_mask = b_mask
        
        #how many mask elements
        a_nr = self.a_mask.sum()
        b_nr = self.b_mask.sum()
        #print (a_nr,b_nr)

        #make distributions with the right number of elements for each re and im mode.
        a_shape = (a_nr)
        loc = torch.zeros(a_shape)
        var = torch.ones(a_shape)
        self.a_dist = torch.distributions.normal.Normal(torch.flatten(loc), torch.flatten(var))
        b_shape = (b_nr)
        loc = torch.zeros(b_shape)
        var = torch.ones(b_shape)
        self.b_dist = torch.distributions.normal.Normal(torch.flatten(loc), torch.flatten(var))
        
        #estimate scalar fudge factor to make unit variance.
        self.rescale = 1.
        samples = self.sample_n(10000)
        self.rescale = 1./np.std(utilities.grab(samples))
                       
    def get_lxly(self, nx, dx):
        """ returns the (lx, ly) pair associated with each Fourier mode. """
        return np.meshgrid( np.fft.fftfreq( nx, dx )[0:int(nx/2+1)]*2.*np.pi,np.fft.fftfreq( nx, dx )*2.*np.pi ) 

    def get_ell(self,nx, dx):
        """ returns the wavenumber l = \sqrt(lx**2 + ly**2) for each Fourier mode """
        lx, ly = self.get_lxly(nx, dx)
        return np.sqrt(lx**2 + ly**2)    
        
    def log_prob(self, x):
        #ignore constant factors
        
        #fft to get the modes
        fft = torch.fft.rfftn(x,dim=[1,2]) * np.sqrt(2.)
        fft[:] *= self.clinvfactor / self.rescale
        x = torch.view_as_real(fft)  
        
        #naive: ignore symmetries
        #logp = self.dist.log_prob(x.reshape(x.shape[0], -1)) 
        #logp = torch.sum(logp, dim=1)
        
        #correct: use symmetries
        a = x[:,:,:,0]
        b = x[:,:,:,1]
        amasked = a[:,self.a_mask]
        bmasked = b[:,self.b_mask]
        logp_a = self.a_dist.log_prob(amasked) 
        logp_b = self.b_dist.log_prob(bmasked)
        logp = torch.sum(logp_a, dim=1) + torch.sum(logp_b, dim=1)
        
        return logp
    
    def sample_n(self, batch_size):
        #https://pytorch.org/docs/stable/complex_numbers.html
        
        #draw random rfft modes
        x = self.dist.sample((batch_size,))
        
        #test logp
        #logptemp = self.dist.log_prob(x.reshape(x.shape[0], -1))
        #print("logp temp", torch.sum(logptemp, dim=1))
        
        #reshape to rfft format
        x = x.reshape(batch_size, *self.rfourier_shape)
        #make complex data type
        fft = torch.view_as_complex(x) / np.sqrt(2.)
         
        #enforce rfft constraints
        #from quicklens
        fft[:,0,0] = np.sqrt(2.) * fft[:,0,0].real #fft.real
        fft[:,int(self.nx/2+1):, 0] = torch.conj( torch.flip(fft[:,1:int(self.nx/2),0], (1,)) ) 
        
        #extra symmetries (assuming th rfft output format is as in numpy)
        fft[:,0,int(self.nx/2)] = fft[:,0,int(self.nx/2)].real * np.sqrt(2.)
        fft[:,int(self.nx/2),0] = fft[:,int(self.nx/2),0].real * np.sqrt(2.)
        fft[:,int(self.nx/2),int(self.nx/2)] = fft[:,int(self.nx/2),int(self.nx/2)].real * np.sqrt(2.)
        fft[:,int(self.nx/2+1):, int(self.nx/2)] = torch.conj( torch.flip(fft[:,1:int(self.nx/2),int(self.nx/2)], (1,)) ) 
        #flip from https://github.com/pytorch/pytorch/issues/229
        #https://pytorch.org/docs/stable/generated/torch.flip.html#torch.flip
        
        #TODO: check normalization of irfftn. see quicklens maps.py line 907 and the new options of irfftn.
        #https://pytorch.org/docs/1.7.1/fft.html#torch.fft.irfftn
        #for now scalar fudge factor to make unit variance.
        
        #adjust mode amplitude to power spectrum
        fft[:] *= self.clfactor * self.rescale
        
        #transform to position space
        rmap = torch.fft.irfftn(fft,dim=[1,2])
        #https://pytorch.org/docs/1.7.1/fft.html#torch.fft.irfftn
        
        return rmap
    
    
    
######################## Flow generalities    
    
    
def apply_flow_to_prior(prior, coupling_layers, batch_size):
    #draws from the prior (base distribution) and flows them
    u = prior.sample_n(batch_size)
    log_pu = prior.log_prob(u)
    z = u.clone()
    log_pz = log_pu.clone()
    for layer in coupling_layers:
        z, logJ = layer.forward(z)
        log_pz = log_pz - logJ
    return u, log_pu, z, log_pz




def apply_reverse_flow_to_sample(z, prior, coupling_layers):
    #takes samples and calculates their representation in base distribution space and their density 
    log_J_Tinv = 0
    n_layers = len(coupling_layers)
    for layer_id in reversed(range(n_layers)):
        layer = coupling_layers[layer_id]
        z, logJ = layer.reverse(z)
        log_J_Tinv = log_J_Tinv + logJ 
    u = z
    log_pu = prior.log_prob(u)
    return u, log_pu, log_J_Tinv


############################################################# Glow


#device = "cuda:0"

def compute_same_pad(kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if isinstance(stride, int):
        stride = [stride]

    assert len(stride) == len(
        kernel_size
    ), "Pass kernel size and stride both as int, or both as equal length iterable"

    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]

def uniform_binning_correction(x, n_bits=8):
    b, c, h, w = x.size()
    n_bins = 2 ** n_bits
    chw = c * h * w
    x += torch.zeros_like(x).uniform_(0, 1.0 / n_bins)

    objective = -math.log(n_bins) * chw * torch.ones(b, device=x.device)
    return x, objective

def split_feature(tensor, type="split"):
    #type = ["split", "cross"]
    C = tensor.size(1)
    if type == "split":
        return tensor[:, : C // 2, ...], tensor[:, C // 2 :, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]

def gaussian_p(mean, logs, x):
    c = math.log(2 * math.pi)
    return -0.5 * (logs * 2.0 + ((x - mean) ** 2) / torch.exp(logs * 2.0) + c)

def gaussian_likelihood(mean, logs, x):
    p = gaussian_p(mean, logs, x)
    return torch.sum(p, dim=[1, 2, 3])

def gaussian_sample(mean, logs, temperature=1):
    # Sample from Gaussian with temperature
    z = torch.normal(mean, torch.exp(logs) * temperature)

    return z

def squeeze2d(input, factor):
    if factor == 1:
        return input

    B, C, H, W = input.size()
    
    assert H % factor == 0 and W % factor == 0, "H or W modulo factor is not 0"

    x = input.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)

    return x

def unsqueeze2d(input, factor):
    if factor == 1:
        return input

    factor2 = factor ** 2

    B, C, H, W = input.size()

    assert C % (factor2) == 0, "C module factor squared is not 0"

    x = input.view(B, C // factor2, factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor2), H * factor, W * factor)

    return x

class _ActNorm(nn.Module):
    #Initialize the bias and scale with a given minibatch, so that the output per-channel have zero mean and unit variance for that. After initialization, `bias` and `logs` will be trained as parameters.
    def __init__(self, num_features, scale=1.0):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1, 1]
        self.bias = nn.Parameter(torch.zeros(*size))
        self.logs = nn.Parameter(torch.zeros(*size))
        self.num_features = num_features
        self.scale = scale
        self.inited = False

    def initialize_parameters(self, input):
        if not self.training:
            raise ValueError("In Eval mode, but ActNorm not inited")

        with torch.no_grad():
            bias = -torch.mean(input.clone(), dim=[0, 2, 3], keepdim=True)
            vars = torch.mean((input.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))

            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)

            self.inited = True

    def _center(self, input, reverse=False):
        if reverse:
            return input - self.bias
        else:
            return input + self.bias

    def _scale(self, input, logdet=None, reverse=False):

        if reverse:
            input = input * torch.exp(-self.logs)
        else:
            input = input * torch.exp(self.logs)

        if logdet is not None:
            #logs is log_std of `mean of channels` so we need to multiply by number of pixels
            b, c, h, w = input.shape

            dlogdet = torch.sum(self.logs) * h * w

            if reverse:
                dlogdet *= -1

            logdet = logdet + dlogdet

        return input, logdet

    def forward(self, input, logdet=None, reverse=False):
        self._check_input_dim(input)

        if not self.inited:
            self.initialize_parameters(input)

        if reverse:
            input, logdet = self._scale(input, logdet, reverse)
            input = self._center(input, reverse)
        else:
            input = self._center(input, reverse)
            input, logdet = self._scale(input, logdet, reverse)

        return input, logdet

class ActNorm2d(_ActNorm):
    def __init__(self, num_features, scale=1.0):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.size()) == 4
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCHW`,"
            " channels should be {} rather than {}".format(
                self.num_features, input.size()
            )
        )

class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="same",
        do_actnorm=True,
        weight_std=0.05,
    ):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=(not do_actnorm),
        )
        # init weight with std
        self.conv.weight.data.normal_(mean=0.0, std=weight_std)

        if not do_actnorm:
            self.conv.bias.data.zero_()
        else:
            self.actnorm = ActNorm2d(out_channels)

        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = self.conv(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x

class Conv2dZeros(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="same",
        logscale_factor=3,
    ):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

        self.logscale_factor = logscale_factor
        self.logs = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, input):
        output = self.conv(input)
        return output * torch.exp(self.logs * self.logscale_factor)

class Permute2d(nn.Module):
    def __init__(self, num_channels, shuffle):
        super().__init__()
        self.num_channels = num_channels
        self.indices = torch.arange(self.num_channels - 1, -1, -1, dtype=torch.long)
        self.indices_inverse = torch.zeros((self.num_channels), dtype=torch.long)

        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

        if shuffle:
            self.reset_indices()

    def reset_indices(self):
        shuffle_idx = torch.randperm(self.indices.shape[0])
        self.indices = self.indices[shuffle_idx]

        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

    def forward(self, input, reverse=False):
        assert len(input.size()) == 4, print(len(input.size()))

        if not reverse:
            input = input[:, self.indices, :, :]
            return input
        else:
            return input[:, self.indices_inverse, :, :]

class Split2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv = Conv2dZeros(num_channels // 2, num_channels)

    def split2d_prior(self, z):
        h = self.conv(z)
        return split_feature(h, "cross")

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            z1 = input
            mean, logs = self.split2d_prior(z1)
            z2 = gaussian_sample(mean, logs, temperature)
            z = torch.cat((z1, z2), dim=1)
            return z, logdet
        else:
            z1, z2 = split_feature(input, "split")
            mean, logs = self.split2d_prior(z1)
            logdet = gaussian_likelihood(mean, logs, z2) + logdet
            return z1, logdet

class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, logdet=None, reverse=False):
        if reverse:
            output = unsqueeze2d(input, self.factor)
        else:
            output = squeeze2d(input, self.factor)

        return output, logdet

device = 'cuda:0'
    
class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = torch.qr(torch.randn(*w_shape))[0]

        if not LU_decomposed:
            self.weight = nn.Parameter(torch.Tensor(w_init))
        else:
            p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
            s = torch.diag(upper)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            upper = torch.triu(upper, 1)
            l_mask = torch.tril(torch.ones(w_shape), -1)
            eye = torch.eye(*w_shape)

            self.register_buffer("p", p)
            self.register_buffer("sign_s", sign_s)
            self.lower = nn.Parameter(lower)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(upper)
            self.l_mask = l_mask
            self.eye = eye

        self.w_shape = w_shape
        self.LU_decomposed = LU_decomposed

    def get_weight(self, input, reverse):
        b, c, h, w = input.shape

        if not self.LU_decomposed:
            dlogdet = torch.slogdet(self.weight)[1] * h * w
            if reverse:
                weight = torch.inverse(self.weight)
            else:
                weight = self.weight
        else:
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)

            lower = self.lower * self.l_mask + self.eye

            u = self.upper * self.l_mask.transpose(0, 1).contiguous()
            u += torch.diag(self.sign_s * torch.exp(self.log_s))

            dlogdet = torch.sum(self.log_s) * h * w

            if reverse:
                u_inv = torch.inverse(u)
                l_inv = torch.inverse(lower)
                p_inv = torch.inverse(self.p)

                weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
            else:
                weight = torch.matmul(self.p, torch.matmul(lower, u))

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        #log-det = log|abs(|W|)| * pixels
        weight, dlogdet = self.get_weight(input, reverse)
        input = input.to(device=device, dtype=torch.float)
        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet

def get_block(in_channels, out_channels, hidden_channels):
    block = nn.Sequential(
        Conv2d(in_channels, hidden_channels),
        nn.ReLU(inplace=False),
        Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1)),
        nn.ReLU(inplace=False),
        Conv2dZeros(hidden_channels, out_channels),
    )
    return block

class FlowStep(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
    ):
        super().__init__()
        self.flow_coupling = flow_coupling

        self.actnorm = ActNorm2d(in_channels, actnorm_scale)
        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=LU_decomposed)
            self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)
        elif flow_permutation == "shuffle":
            self.shuffle = Permute2d(in_channels, shuffle=True)
            self.flow_permutation = lambda z, logdet, rev: (
                self.shuffle(z, rev),
                logdet,
            )
        else:
            self.reverse = Permute2d(in_channels, shuffle=False)
            self.flow_permutation = lambda z, logdet, rev: (
                self.reverse(z, rev),
                logdet,
            )
        # 3. coupling
        if flow_coupling == "additive":
            self.block = get_block(in_channels // 2, in_channels // 2, hidden_channels)
        elif flow_coupling == "affine":
            self.block = get_block(in_channels // 2, in_channels, hidden_channels)

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, logdet)
        else:
            return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):
        assert input.size(1) % 2 == 0
        # 1. actnorm
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)
        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, False)
        # 3. coupling
        z1, z2 = split_feature(z, "split")
        if self.flow_coupling == "additive":
            z2 = z2 + self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 + shift
            z2 = z2 * scale
            if logdet == None: ################### Cannot + if logdet = None
                logdet = torch.sum(torch.log(scale), dim=[1, 2, 3])
            else:
                logdet = torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        return z, logdet

    def reverse_flow(self, input, logdet):
        assert input.size(1) % 2 == 0
        # 1.coupling
        z1, z2 = split_feature(input, "split")
        if self.flow_coupling == "additive":
            z2 = z2 - self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)
        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, True)
        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet

class FlowNet(nn.Module):
    def __init__(
        self,
        image_shape,
        hidden_channels,
        K,
        L,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.output_shapes = []

        self.K = K
        self.L = L

        H, W, C = image_shape
        
        learn_top = False
        self.learn_top = learn_top
        # learned prior
        if learn_top:
            C = self.flow.output_shapes[-1][1]
            self.learn_top_fn = Conv2dZeros(C * 2, C * 2)

        for i in range(L):
            # 1. Squeeze
            C, H, W = C * 4, H // 2, W // 2
            self.layers.append(SqueezeLayer(factor=2))
            self.output_shapes.append([-1, C, H, W])
            # 2. K FlowStep
            for _ in range(K):
                self.layers.append(
                    FlowStep(
                        in_channels=C,
                        hidden_channels=hidden_channels,
                        actnorm_scale=actnorm_scale,
                        flow_permutation=flow_permutation,
                        flow_coupling=flow_coupling,
                        LU_decomposed=LU_decomposed,
                    )
                )
                self.output_shapes.append([-1, C, H, W])
            # 3. Split2d
            if i < L - 1:
                self.layers.append(Split2d(num_channels=C))
                self.output_shapes.append([-1, C // 2, H, W])
                C = C // 2
                
        self.register_buffer(
            "prior_h",
            torch.zeros(
                [
                    1,
                    self.output_shapes[-1][1] * 2,
                    self.output_shapes[-1][2],
                    self.output_shapes[-1][3],
                ]
            ),
        )

    def prior(self, data):
        if data is not None:
            h = self.prior_h.repeat(data.shape[0], 1, 1, 1)
        else:
            h = self.prior_h.repeat(100, 1, 1, 1)

        if self.learn_top:
            h = self.learn_top_fn(h)

        return split_feature(h, "split")

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            return self.reverse_flow(input, nx, temperature)
        else:
            return self.normal_flow(input)

    def encode(self, z, logdet=0.0):
        for layer, shape in zip(self.layers, self.output_shapes):
            z, logdet = layer(z, logdet, reverse=False)
        return z, logdet

    def normal_flow(self, x):
        b, c, h, w = x.shape

        x, logdet = uniform_binning_correction(x)

        z, objective = self.encode(x, logdet=logdet)

        mean, logs = self.prior(x)
        
        log_pu = (gaussian_likelihood(mean,logs,z))   #removed / (math.log(2.0) * c * h * w)
        log_J_Tinv = (objective)                      #removed / (math.log(2.0) * c * h * w)

        return z, log_pu, log_J_Tinv       

    def decode(self, z, temperature=None):
        for layer in reversed(self.layers):
            if isinstance(layer, Split2d):
                z, logdet = layer(z, logdet=0, reverse=True, temperature=temperature)
            else:
                z, logdet = layer(z, logdet=0, reverse=True)
        return z

    def reverse_flow(self, z, temperature):
        mean, logs = self.prior(z)
        with torch.no_grad():
            if z is None:
                mean, logs = self.prior(z)
                z = gaussian_sample(mean, logs, temperature)
            x = self.decode(z, temperature=temperature)
        return x, logs

    def set_actnorm_init(self):
        for name, m in self.named_modules():
            if isinstance(m, ActNorm2d):
                m.inited = True
