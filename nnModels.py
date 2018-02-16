import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as ag
from torch.optim.lr_scheduler import ReduceLROnPlateau
from AutoEncoders import AutoEncoder as AE
import numpy as np


class NN_Manage():
    """
    This is a class to manage the training, test and prediction of a neural network.  All of the things it does
    could be done separately but this makes it less hassle
    """
    def __init__(self, model, loss_function, optimizer, scheduler, HAS_CUDA=False, my_logger=False):
        self.model=model
        self.loss_function=loss_function
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.HAS_CUDA=HAS_CUDA
        self.my_logger=my_logger
        self.train_log=[]
        self.test_log=[]

    def to_np(self, x):
        if self.HAS_CUDA:
            return x.data.cpu().numpy()
        else:
            return x.data.numpy()

    def train(self,epoch, shuffle, inputs, targets, batch_size, train_log=False):
        tr_batches=int(inputs.shape[0]/batch_size)

        if shuffle:
            p = torch.from_numpy(np.random.permutation(tr_batches * batch_size))
            inputs_tr = inputs[p]
            targets_tr = targets[p]
        else:
            inputs_tr = inputs
            targets_tr = targets
        self.model.train()
        train_loss = 0
        for batch in range(tr_batches):
            batch_data = inputs_tr[batch * batch_size:(batch + 1) * batch_size]
            target_data = targets_tr[batch * batch_size:(batch + 1) * batch_size]
            self.optimizer.zero_grad()
            t_data = ag.Variable(batch_data)
            t_target = ag.Variable(target_data, requires_grad=False)
            if self.HAS_CUDA:
                    t_data=t_data.cuda()
                    t_target=t_target.cuda()
            pred = self.model.forward(t_data)
            loss = self.loss_function(pred, t_target)
            loss.backward()
            train_loss += loss.data[0]
            if self.my_logger:
                self.train_log.append(self.to_np(loss[0]))
            self.optimizer.step()
            if batch % 100 ==0:
                if epoch==0:
                    if batch % 100 == 0:
                        print('Train Epoch: {}  Batch: {} Loss: {:.6f}'.format(
                            epoch, batch, loss.data[0]))

        if epoch % 1 ==0:
            print('====> Epoch: {} Average training loss:      {:.4f}'.format(
                epoch, train_loss / tr_batches))

        return train_loss / tr_batches

    def test(self, epoch, inputs, targets, batch_size):
        val_batches=int(inputs.size()[0]/batch_size)
        self.model.eval()
        test_loss = 0
        for batch in range(val_batches):
            batch_data = inputs[batch * batch_size:(batch + 1) * batch_size]
            target_data = targets[batch * batch_size:(batch + 1) * batch_size]
            v_data = ag.Variable(batch_data)
            v_target = ag.Variable(target_data)
            if self.HAS_CUDA:
                    v_data=v_data.cuda()
                    v_target=v_target.cuda()
            pred = self.model.forward(v_data)
            loss = self.loss_function(pred, v_target)
            test_loss += loss.data[0]
            if self.my_logger:
                self.test_log.append(self.to_np(loss[0]))

        test_loss /= val_batches

        if epoch % 1 ==0:
            print('====> Epoch: {} Average validation loss:          {:.4f}'.format(epoch, test_loss))
        return test_loss

    def predict(self, inputs, batch_size, method):
        """
        Carry out a forward pass throught the trained model.  Makes sure that all cases are passed by padding the
        last batch if necessary

        """
        no_inputs=inputs.size()[0]
        batches=int(no_inputs/batch_size)
        remain=no_inputs-batches*batch_size
        first_pass=True
        self.model.eval()
        for batch in range(batches):
            batch_data = inputs[batch * batch_size:(batch + 1) * batch_size]
            data = ag.Variable(batch_data)
            if self.HAS_CUDA:
                data=data.cuda()
            pred = method(data)
            pred = pred.data
            if self.HAS_CUDA:
                pred=pred.cpu()
            if first_pass:
                predict = torch.FloatTensor(inputs.shape[0], pred.shape[1]).zero_()
                first_pass=False
            predict[batch * batch_size:(batch + 1) * batch_size]=pred
        if remain > 0:
            data=ag.Variable(inputs[batches*batch_size:,:])
            if self.HAS_CUDA:
                data=data.cuda()
            pred = method(data)
            pred = pred.data
            if self.HAS_CUDA:
                pred=pred.cpu()
            predict[batches*batch_size:,:]=pred
        return predict


class SmClassify(nn.Module):
    """
    Simple classifier intended to sit at the end of a network and to be used with NNNLoss function.
    This model has no hidden layer
    """
    def __init__(self, n_in, n_classes):
        super(SmClassify, self).__init__()
        self.n_in=n_in
        self.n_classes=n_classes
        self.lin1=nn.Linear(n_in,n_classes)
        self.Sm=nn.LogSoftmax()

    def forward(self, input):
        x=self.lin1(input)
        x=self.Sm(x)
        return x


class Combined_Classifier(nn.Module):
    def __init__(self, l1_model, l2_model, sm_model):
        super(Combined_Classifier, self).__init__()

        # Copy relevant parts of models to a new model
        self.l1_model=l1_model
        self.l2_model=l2_model
        self.sm_model=sm_model

        self.l1 = nn.Linear(l1_model.n_outer, l1_model.n_hid1)
        self.l1_activation = l1_model.enc_activation
        self.l2 = nn.Linear(l2_model.n_outer, l2_model.n_hid1)
        self.l2_activation = l2_model.enc_activation
        self.Sm_lin=nn.Linear(sm_model.n_in,sm_model.n_classes)
        self.Sm_sm=nn.LogSoftmax()

    def forward(self, inputs):
        x=inputs.view(-1, self.l1_model.n_outer)
        x=self.l1(x)
        x=self.l1_activation(x)
        # Model 2 layer
        x=self.l2(x)
        x=self.l2_activation(x)
        x# Model 3 layer
        x=self.Sm_lin(x)
        x=self.Sm_sm(x)
        return x

class Combined_Classifier_HL(nn.Module):
    """
    This classifier includes a hidden layer in the softmax algorithm
    """
    def __init__(self, l1_model, l2_model, sm_model):
        super(Combined_Classifier_HL, self).__init__()

        # Copy relevant parts of models to a new model
        self.l1_model=l1_model
        self.l2_model=l2_model
        self.sm_model=sm_model

        self.l1 = nn.Linear(l1_model.n_outer, l1_model.n_hid1)
        self.l1_activation = l1_model.enc_activation
        self.l2 = nn.Linear(l2_model.n_outer, l2_model.n_hid1)
        self.l2_activation = l2_model.enc_activation
        self.hid1 = nn.Linear(sm_model.n_in, sm_model.n_hid)
#        self.do = nn.Dropout(p=0.5)
        self.Sm_lin = nn.Linear(sm_model.n_hid, sm_model.n_classes)
        self.Sm_sm = nn.LogSoftmax()

    def forward(self, inputs):
        x=inputs.view(-1, self.l1_model.n_outer)
        x=self.l1(x)
        x=self.l1_activation(x)
        # Model 2 layer
        x=self.l2(x)
        x=self.l2_activation(x)
        x=self.hid1(x)
        x = F.leaky_relu(x)
        x=self.Sm_lin(x)
        x=self.Sm_sm(x)
        return x

class Combined_AE(nn.Module):
    def __init__(self, l1_model, l2_model):
        super(Combined_AE, self).__init__()

        # Copy relevant parts of models to a new model
        self.l1_model=l1_model
        self.l2_model=l2_model

        self.enc_l1 = nn.Linear(l1_model.n_outer, l1_model.n_hid1)
        self.dec_l1 = nn.Linear(l1_model.n_hid1, l1_model.n_outer)
        self.l1_enc_activation = l1_model.enc_activation
        self.l1_dec_activation=l1_model.dec_activation

        self.enc_l2 = nn.Linear(l2_model.n_outer, l2_model.n_hid1)
        self.dec_l2 = nn.Linear(l2_model.n_hid1, l2_model.n_outer)
        self.l2_enc_activation = l2_model.enc_activation
        self.l2_dec_activation=l2_model.dec_activation

    def encoder(self, inputs):
        # Encode through two layers
        x=inputs.view(-1, self.l1_model.n_outer)
        x=self.enc_l1(x)
        x=self.l1_enc_activation(x)
        # Model 2 layer
        x=self.enc_l2(x)
        x=self.l2_enc_activation(x)
        return x

    def decoder(self, x):
        x=self.dec_l2(x)
        if self.l2_dec_activation:
            x=self.l2_enc_activation(x)
        x=self.dec_l1(x)
        if self.l1_dec_activation:
            x=self.l1_enc_activation(x)
        return x

    def forward(self, inputs):
        # Encode through two layers
        x=self.encoder(inputs)
        # Decode through two layers
        x=self.decoder(x)
        return x

class Combined_Regression_HL(nn.Module):
    """
    This classifier includes a hidden layer in the softmax algorithm
    """
    def __init__(self, l1_model, l2_model, n_hid, n_lin):
        super(Combined_Regression_HL, self).__init__()

        # Copy relevant parts of models to a new model
        self.l1_model=l1_model
        self.l2_model=l2_model
        self.n_hid=n_hid
        self.n_lin=n_lin

        self.l1 = nn.Linear(l1_model.num_visible, l1_model.num_hidden)
        self.l1_activation = F.sigmoid
        self.l2 = nn.Linear(l2_model.num_visible, l2_model.num_hidden)
        self.l2_activation = F.sigmoid
        self.hid = nn.Linear(l2_model.num_hidden, n_hid)
        self.hid_activation=F.sigmoid
        self.lin = nn.Linear(n_hid, n_lin)
        self.out=nn.Linear(n_lin,1)
#        self.do = nn.Dropout(p=0.5)

    def forward(self, inputs):
        x=inputs.view(-1, self.l1_model.num_visible)
        # First RBM
        x=self.l1(x)
        x=self.l1_activation(x)
        # Second RBM
        x=self.l2(x)
        x=self.l2_activation(x)
        # Non-linear regression layer
        x=self.hid(x)
        x = self.hid_activation(x)
        x=self.lin(x)
        x=self.out(x)
        return x