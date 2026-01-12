import torch
from torch.nn import MSELoss
from model import residual_rnn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

def iter_generate(model:residual_rnn,
                  itera_len:int,
                  xt,ht,difft):
    '''
    inputs:
        xt: batch_size x 1 x (itera_dim + fixed_dim)
        ht: num_layer x batch_size x h_dim
        difft: batch_size x 1 x itera_dim
    outputs:
        xt_gen: batch_size x itera_len x (itera_dim + fixed_dim)
    '''
    batch_size = xt.shape[0]
    xt_gen = torch.zeros(size=(batch_size,itera_len,model.itera_dim+model.fixed_dim))

    for i in range(itera_len):
        with torch.no_grad():
            xt,ht,difft = model.one_step_forward(xt,ht,difft)
            xt_squeeze = xt.squeeze(1)
            xt_gen[:,i,:] = xt_squeeze
    
    return xt_gen

def full_generate(model:residual_rnn,
                  input,
                  label_len):
    
    with torch.no_grad():
        diff, ht = model.long_seq_forward(input)

        xt = input[:,-1,:]
        xt = xt.unsqueeze(1)

        difft = diff[:,-1,:]
        difft = difft.unsqueeze(1)

        itera_length = label_len - input.shape[1]

        xt_gen = iter_generate(model=model,
                               itera_len=itera_length,
                               xt=xt,
                               ht=ht,
                               difft=difft)
        
        xt_gen = xt_gen.to('cuda')
        xt_full = torch.concat((input,xt_gen),dim=1)
    
    return xt_full

def train_and_valid(model:residual_rnn,
                    dataloader:DataLoader,
                    loss_fn:MSELoss,
                    mode:str,
                    optimizer:Optimizer=None):
    
    if mode == 'train':
        model.train()
    elif mode == 'valid':
        model.eval()

    total_loss = 0

    for input,label in dataloader:
        input = input.to('cuda')
        label = label.to('cuda')

        label_len = label.shape[1]
        xt_full = full_generate(model,input,label_len)

        if mode == 'train':
            diff,_ = model.long_seq_forward(xt_full)

            optimizer.zero_grad()
            loss = loss_fn(diff,label)
            loss.backward()
            optimizer.step()
        
        elif mode == 'valid':
            with torch.no_grad():
                diff,_ = model.long_seq_forward(xt_full)
                loss = loss_fn(diff,label)

        total_loss += loss.item()
    
    total_loss /= len(dataloader.dataset)
    return total_loss