import torch
import numpy as np
import torch.nn as nn

class residual_rnn(nn.Module):
    def __init__(self,
                 h_dim:int,
                 itera_dim=7,
                 fixed_dim=5):
        super().__init__()

        torch.set_default_dtype(torch.float64)

        self.itera_dim = itera_dim
        self.fixed_dim = fixed_dim

        self.rnn = nn.RNN(input_size=itera_dim,
                          hidden_size=h_dim,
                          num_layers=1,
                          batch_first=True)
        
        # automatically generate fc_V
        layers = []
        expon_ul = int(np.floor(np.log2(h_dim)))
        expon_dl = int(np.ceil(np.log2(itera_dim)))

        current_dim = h_dim+fixed_dim
        for i in range(expon_ul,expon_dl-1,-1):
            layer_dim = 2**i
            layers.append(nn.Linear(in_features=current_dim,
                                    out_features=layer_dim))
            layers.append(nn.LeakyReLU(0.01))
            current_dim = layer_dim
        
        layers.append(nn.Linear(in_features=current_dim,
                                out_features=itera_dim))
        
        self.fc_V = nn.Sequential(*layers)
        return None
    
    def long_seq_forward(self,x):
        '''
        inputs:
            x: batch_size x input_len x (itera_dim + fixed_dim)
        outputs:
            ot: batch_size x input_len x itera_dim
            ht_last: num_layer x batch_size x h_dim
        '''
        x_itera = x[:,:,:self.itera_dim]
        x_fixed = x[:,:,self.itera_dim:]
        # x_fixed: batch_size x input_len x fixed_dim

        ht, ht_last = self.rnn(x_itera)
        # ht: batch_size x input_len x h_dim
        # ht_lase: num_layer x batch_size x h_dim

        ht_full = torch.concat((ht,x_fixed),dim=-1)
        ot = self.fc_V(ht_full)

        return ot,ht_last
    
    def one_step_forward(self,xt,ht,difft):
        '''
        inputs:
            xt: batch_size x 1 x (itera_dim + fixed_dim)
            ht: num_layer x batch_size x h_dim
            difft: batch_size x 1 x itera_dim
        outputs:
            xtp1, htp1, difftp1: all same size as corresponding input
        '''

        x_itera = xt[:,:,:self.itera_dim]
        x_fixed = xt[:,:,self.itera_dim:]

        xtp1 = x_itera + difft # batch_size x 1 x itera_dim
        htp1_batch_first, htp1 = self.rnn(xtp1,ht)

        htp1_full = torch.concat((htp1_batch_first,x_fixed),dim=-1)
        difftp1 = self.fc_V(htp1_full)

        xtp1 = torch.concat((xtp1,x_fixed),dim=-1)
        return xtp1, htp1, difftp1