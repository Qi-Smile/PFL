from alg.fedavg import fedavg
from util.modelsel import modelsel
import torch.optim as optim
import torch.nn as nn
from util.traineval import train, test, train_moe, test_moe, train_prox
from alg.core.comm import communication
import copy

class fedmoe(fedavg):
    def __init__(self,args):
        super(fedmoe, self).__init__(args)
        self.loss_fun = nn.CrossEntropyLoss()
        self.private_model = copy.deepcopy(self.client_model)
        self.private_optimizers = [optim.Adam(params=self.private_model[idx].parameters(
        ), lr=0.001) for idx in range(args.n_clients)]
        self.shared_model = copy.deepcopy(self.client_model)
        self.args = args
 
    def client_train(self, c_idx, dataloader, round):
        for _ in range(self.args.shared_epoch):
             if round > 0:
                 train_prox(
                self.args, self.client_model[c_idx], self.server_model, dataloader, self.optimizers[c_idx], self.loss_fun, self.args.device)
             else:
                train(
                    self.client_model[c_idx], dataloader, self.optimizers[c_idx], self.loss_fun, self.args.device)
        for _ in range(self.args.private_epoch):
            train_loss, train_acc = train_moe(self.args, self.shared_model[c_idx], self.private_model[c_idx], dataloader, self.private_optimizers[c_idx], self.loss_fun, self.args.device)
        if round % self.args.update_epoch == 0 and round>0:
            self.shared_model = copy.deepcopy(self.client_model)
        return train_loss, train_acc

    def server_aggre(self):
        self.server_model, self.client_model = communication(
            self.args, self.server_model, self.client_model, self.client_weight)

    def client_eval(self, c_idx, dataloader):
        train_loss, train_acc = test_moe(
            self.args, self.shared_model[c_idx], self.private_model[c_idx], dataloader, self.loss_fun, self.args.device)
        # train_loss, train_acc = test(
        #     self.client_model[c_idx], dataloader, self.loss_fun, self.args.device)
        return train_loss, train_acc

    def server_eval(self, dataloader):
        train_loss, train_acc = test(
            self.server_model, dataloader, self.loss_fun, self.args.device)
        return train_loss, train_acc
        
   