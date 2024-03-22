# coding=utf-8
from alg.fedavg import fedavg
from util.traineval import train, train_prox


class fedprox(fedavg):
    def __init__(self, args):
        super(fedprox, self).__init__(args)

    def client_train(self, c_idx, dataloader, round):
        # prox只有round>0的时候可以调用train_prox，也就意味着round=0的初始阶段应该是本地训练的
        if round > 0:
            train_loss, train_acc = train_prox(
                self.args, self.client_model[c_idx], self.server_model, dataloader, self.optimizers[c_idx], self.loss_fun, self.args.device)
        else:
            train_loss, train_acc = train(
                self.client_model[c_idx], dataloader, self.optimizers[c_idx], self.loss_fun, self.args.device)
        return train_loss, train_acc
