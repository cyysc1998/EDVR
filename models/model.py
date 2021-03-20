import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import sys
sys.path.append('./models')
from dgcnn import DGCNN
from tools import MLP


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used 
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b).type_as(x)
        running_var = self.running_var.repeat(b).type_as(x)
        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


# DenseNet Decoder

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", AdaptiveInstanceNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv1d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))

        self.add_module("norm2", AdaptiveInstanceNorm2d(bn_size * growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv1d(bn_size * growth_rate, growth_rate, kernel_size=1, stride=1, bias=False))
    
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x1 = super(_DenseLayer, self).forward(x)
        x = torch.cat([x, x1], dim=1)

        return x

class DenseNet(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate):
        super(DenseNet, self).__init__()
        self.model = nn.Sequential()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size)
            self.model.add_module("Denselayer%d" % (i), layer)
    def forward(self, x):
        x = self.model(x)
        return x

# Model

class ConsNet(nn.Module):
    def __init__(self, args, seg_num_all):
        super(ConsNet, self).__init__()
        self.dgcnn = DGCNN(args, seg_num_all)
        self.decoder = DenseNet(args.nlayers, 256, 4, 128)
        
        self.output_dim = []
        for i in range(args.nlayers):
            self.output_dim.append(256 + 128 * i)
            self.output_dim.append(512)
        self.mlp_w = MLP(2*args.num_points, 64 * args.nlayers * args.nlayers + 704 * args.nlayers, 256, 3)
        self.mlp_b = MLP(2*args.num_points, 64 * args.nlayers * args.nlayers + 704 * args.nlayers, 256, 3)

        self.conv = nn.Conv1d(256 + 128 * args.nlayers, 3, kernel_size=1)

    def forward(self, x, y, l):
        x = self.dgcnn(x, l) # (B, 256, N)
        y = y.permute(0, 2, 1).reshape(y.size(0), -1)
      
        adain_params_w = self.mlp_w(y)
        adain_params_b = self.mlp_b(y)
        self.assign_adain_params(adain_params_w, adain_params_b, self.decoder)
        
        x = self.decoder(x)
        x = self.conv(x)

        return x
        
    def assign_adain_params(self, adain_params_w, adain_params_b, model):
        # assign the adain_params to the AdaIN layers in model
        dim = self.output_dim
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params_b[:,:dim[0]].contiguous()
                std = adain_params_w[:,:dim[0]].contiguous()
                m.bias = mean.view(-1)
                m.weight = std.view(-1)
                if adain_params_w.size(1)>dim[0] :  #Pop the parameters
                    adain_params_b = adain_params_b[:,dim[0]:]
                    adain_params_w = adain_params_w[:,dim[0]:]
                    dim = dim[1:]
