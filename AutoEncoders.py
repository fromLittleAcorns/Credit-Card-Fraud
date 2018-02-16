#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Note - I intend to add l1 weighting to create a sparse encoder and then noise
to create a de-noised decoder.  The first encoder is a very simple one layer one
with none of the above.

Created on Tue Aug 15 14:46:13 2017

@author: johnrichmond
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

def default_activation(x):
        return F.tanh(x)

class AutoEncoder(nn.Module):
    def __init__(self, n_outer, n_hid, activation=default_activation,decoder='linear'):
        super(AutoEncoder, self).__init__()
        self.n_outer=n_outer
        self.n_hid1=n_hid
        self.enc_l1=nn.Linear(n_outer,n_hid)
        self.dec_l1=nn.Linear(n_hid, n_outer)
        self.enc_activation=activation
        if decoder=='linear':
            self.dec_activation=False
        else:
            self.dec_activation=True

    def encoder(self,input):
        x=input.view(-1, self.n_outer)
        x=self.enc_l1(x)
        #h1=F.tanh(hid)
        h1=self.enc_activation(x)
        return h1
        
    def decoder(self,input):
        x=input.view(-1, self.n_hid1)
        #out=F.tanh(self.dec_l1(x))
        if self.dec_activation:
            out=self.enc_activation(self.dec_l1(x))
        else:
            out=self.dec_l1(x)
        return out
        
    def forward(self,input):
        x=self.encoder(input)
        x=self.decoder(x)
        return x.view_as(input)


class AutoEncoder_Multi_Layer(nn.Module):
    def __init__(self, n_outer, hidden_sizes, activation=default_activation):
        super(AutoEncoder_Multi_Layer, self).__init__()
        self.num_layers=len(hidden_sizes)
        self.last_encoder=self.num_layers*2
        self.n_outer = n_outer
        self.n_min=hidden_sizes[-1]
        self.activation=nn.LeakyReLU()

        # Create hidden layers
        self.layers=nn.ModuleList([])
        # Create encoding layers
        new_layer=nn.Linear(n_outer, hidden_sizes[0])
        self.layers.append(new_layer)
        new_layer=nn.LeakyReLU()
        new_layer = self.activation
        self.layers.append(new_layer)
        for layer in range(1,self.num_layers):
            new_layer=nn.Linear(hidden_sizes[layer-1], hidden_sizes[layer])
            self.layers.append(new_layer)
            new_layer = self.activation
            self.layers.append(new_layer)
        # Completed encoding layers
        # Create decoding layers
        for layer in range(self.num_layers,1,-1):
            new_layer=nn.Linear(hidden_sizes[layer-1], hidden_sizes[layer-2])
            self.layers.append(new_layer)
            new_layer = self.activation
            self.layers.append(new_layer)
        new_layer=nn.Linear(hidden_sizes[0], n_outer)
        self.layers.append(new_layer)
        # No activation for last layer to allow for scaling
        # Completed decoding layers

    def encoder(self, input):
        x = input.view(-1, self.n_outer)
        for i in range(0,self.last_encoder):
            x=self.layers[i](x)
        return x

    def decoder(self, input):
        x = input.view(-1, self.n_min)
        for i in range(self.last_encoder,len(self.layers)):
            x=self.layers[i](x)
        return x

    def forward(self, input):
        x = self.encoder(input)
        x = self.decoder(x)
        return x.view_as(input)

from torch.autograd import Function

class L1Penalty(Function):

    @staticmethod
    def forward(ctx, input, l1weight):
        ctx.save_for_backward(input)
        ctx.l1weight = l1weight
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        grad_input = input.clone().sign().mul(ctx.l1weight)
        grad_input += grad_output
        return grad_input 
    
class SparseAutoEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, l1weight,
                 activation=default_activation):
        super(SparseAutoEncoder, self).__init__()
        self.n_outer=feature_size
        self.n_hid1=hidden_size
        self.enc_l1=nn.Linear(self.n_outer,self.n_hid1)
        self.dec_l1=nn.Linear(self.n_hid1, self.n_outer) 
        self.l1weight = l1weight
        self.activation=activation
        
    def encoder(self,input):
        x=input.view(-1, self.n_outer)
        h1=self.activation(self.enc_l1(x))
        return h1
    
    def decoder(self,input):
        x=input.view(-1, self.n_hid1)
        out=self.activation(self.enc_l1(x))
        return out
    
    def forward(self,input):
        x=self.encode(input)
        x=L1Penalty.apply(x,self.l1weight)
        x=self.decode(input)
        return x.view_as(input)

class RBM(nn.Module):
    """
    Class to implement CD process for Restricted Boltzman Machine
        n_vis = nodes for input layer
        n_hid = nodes for hidden layer
        k = number of steps of CD process to go through
        use_prob_vis = Instead of using value of V use the probability of V as the input to the next calc
        use_prob_hid =  As above
    """
    def __init__(self,
                 n_vis=784,
                 n_hid=500,
                 k=5,
                 use_prob_vis=True,
                 use_prob_hid=True):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hid, n_vis) * 1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hid))
        self.k = k
        self.use_prob_vis = use_prob_vis
        self.use_prob_hid = use_prob_hid

    def sample_from_p(self, p):
        return F.relu(torch.sign(p - Variable(torch.rand(p.size()))))
        # Note - torch.rand(p.size()) returns an array of random numbers of th esame size as p
        # p-the random number array returns a random number centred on 0.5
        # Hence the returned value will be -1,0 or 1.  The rely then makes this 0 to 1


    def v_to_h(self, v):
        p_h = F.sigmoid(F.linear(v, self.W, self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h

    def h_to_v(self, h):
        p_v = F.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v, sample_v

    def forward(self, v):
        prob_h1, h1 = self.v_to_h(v)
        prob_h_ = prob_h1
        h_ = h1
        for _ in range(self.k):
            if self.use_prob_hid:
                prob_v_, v_ = self.h_to_v(prob_h_)
            else:
                prob_v_, v_ = self.h_to_v(h_)
            if self.use_prob_vis:
                prob_h_, h_ = self.v_to_h(prob_v_)
            else:
                prob_h_, h_ = self.v_to_h(v_)

        return v, v_

    def get_output(self, v):
        prob_h1, h1 = self.v_to_h(v)
        return prob_h1

    def free_energy(self, v):
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()

class RBM_JR():
# Based upon the following model:
#   https://github.com/GabrielBianconi/pytorch-rbm/blob/master/rbm.py

    """
    Class to implement CD process for Restricted Boltzman Machine
        num_visible = nodes for input layer
        num_hidden = nodes for hidden layer
        k = number of steps of CD process to go through

        See the following regarding how to update the bias terms
        https://stats.stackexchange.com/questions/139138/updating-bias-with-rbms-restricted-boltzmann-machines
    """
    def __init__(self, num_visible, num_hidden, k, learning_rate=1e-3, momentum_coefficient=0.5, weight_decay=1e-4,
                 use_cuda=True):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda

        self.weights = torch.randn(num_visible, num_hidden) * 0.1
        self.visible_bias = torch.ones(num_visible) * 0.5
        self.hidden_bias = torch.zeros(num_hidden)

        self.weights_momentum = torch.zeros(num_visible, num_hidden)
        self.visible_bias_momentum = torch.zeros(num_visible)
        self.hidden_bias_momentum = torch.zeros(num_hidden)

        if self.use_cuda:
            self.weights = self.weights.cuda()
            self.visible_bias = self.visible_bias.cuda()
            self.hidden_bias = self.hidden_bias.cuda()

            self.weights_momentum = self.weights_momentum.cuda()
            self.visible_bias_momentum = self.visible_bias_momentum.cuda()
            self.hidden_bias_momentum = self.hidden_bias_momentum.cuda()

    def sample_hidden(self, visible_probabilities):
        hidden_activations = torch.matmul(visible_probabilities, self.weights) + self.hidden_bias
        hidden_probabilities = self._sigmoid(hidden_activations)
        return hidden_probabilities

    def sample_visible(self, hidden_probabilities):
        visible_activations = torch.matmul(hidden_probabilities, self.weights.t()) + self.visible_bias
        visible_probabilities = self._sigmoid(visible_activations)
        return visible_probabilities

    def contrastive_divergence(self, input_data):
        # Positive phase
        positive_hidden_probabilities = self.sample_hidden(input_data)

        random_probabilities = torch.rand(self.num_hidden)

        if self.use_cuda:
            random_probabilities = random_probabilities.cuda()

        positive_hidden_activations = (positive_hidden_probabilities >= random_probabilities)

        positive_associations = torch.matmul(input_data.t(), positive_hidden_probabilities)

        # Negative phase
        negative_hidden_probabilities = positive_hidden_probabilities

        for step in range(self.k):
            negative_visible_probabilities = self.sample_visible(negative_hidden_probabilities)
            negative_hidden_probabilities = self.sample_hidden(negative_visible_probabilities)

        negative_associations = torch.matmul(negative_visible_probabilities.t(), negative_hidden_probabilities)

        # Update parameters
        self.weights_momentum *= self.momentum_coefficient
        self.weights_momentum += (positive_associations - negative_associations)

        self.visible_bias_momentum *= self.momentum_coefficient
        self.visible_bias_momentum += torch.sum(input_data - negative_visible_probabilities, dim=0)

        self.hidden_bias_momentum *= self.momentum_coefficient
        self.hidden_bias_momentum += torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)

        batch_size = input_data.size(0)

        self.weights += self.weights_momentum * self.learning_rate / batch_size
        self.visible_bias += self.visible_bias_momentum * self.learning_rate / batch_size
        self.hidden_bias += self.hidden_bias_momentum * self.learning_rate / batch_size

        self.weights -= self.weights * self.weight_decay  # L2 weight decay

        # Compute reconstruction error
        error = torch.sum((input_data - negative_visible_probabilities) ** 2)

        return error

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))