import os.path as osp
import argparse
import numpy as np
import yaml

import torch
import torch.nn.functional as F
from torch_scatter import scatter
import torch_geometric.transforms as T

from data import Planetoid
from models import return_net
from utils import accuracy, HomophilyRank


def train(epoch, config, data, model, optimizer):
    # train
    model.train()
    optimizer.zero_grad()

    # train by class label
    prob_labels = model(data.x, data.edge_index)
    loss_train = F.nll_loss(prob_labels[data.train_mask], data.y[data.train_mask])
    acc_train, _  = accuracy(prob_labels[data.train_mask], data.y[data.train_mask])

    loss_train.backward()
    optimizer.step()

    # validation
    model.eval()
    prob_labels_val = model(data.x, data.edge_index)
    loss_val = F.nll_loss(prob_labels_val[data.val_mask], data.y[data.val_mask])
    acc_val, _ = accuracy(prob_labels_val[data.val_mask], data.y[data.val_mask])
    
    '''print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()), end=' ')'''

    return loss_val


def test(config, data, model):
    model.eval()
    prob_labels_test = model(data.x, data.edge_index)
    loss_test = F.nll_loss(prob_labels_test[data.test_mask], data.y[data.test_mask])

    top = data.homophily_rank[:500]
    bot = data.homophily_rank[-500:]
    acc, _ = accuracy(prob_labels_test[data.test_mask], data.y[data.test_mask])
    acc_top, _ = accuracy(prob_labels_test[top], data.y[top])
    acc_bot, _ = accuracy(prob_labels_test[bot], data.y[bot])

    print("Test set results:",
          "loss(test)= {:.4f}".format(loss_test.data.item()),
          "accuracy(test)= {:.4f}".format(acc.data.item()),
          "accuracy(top)= {:.4f}".format(acc_top.data.item()),
          "accuracy(bottom)= {:.4f}".format(acc_bot.data.item()))

    # validate weights for step
    '''step_weight = torch.sum(model.lin.weight, dim=0)
    step_idx = 0
    idx = []
    num_layers = [config['hidden'] for _ in range(config.layer)]
    for num_layer in num_layers:
        for _ in range(num_layer):
            idx.append(step_idx)
        step_idx += 1
    idx = torch.tensor(idx).to(step_weight.device)
    step_weight = scatter(step_weight, idx, dim=0, reduce='sum')
    print(step_weight)'''

    return acc


def run(config):
    '''torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False'''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Planetoid(root      = '../data/{}'.format(config['dataset']), 
                        name      = config['dataset'], 
                        split     = config['split'], 
                        transform = eval(config['norm']))
    data = dataset[0].to(device)
    # print(data)

    config['n_feat']  = data.x.size()[1]
    config['n_class'] = torch.max(data.y).data.item() + 1
    model = return_net(config).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = config['learning_rate'], 
                                 weight_decay = config['weight_decay'])

    best_loss = 100.
    bad_counter = 0
    for epoch in range(1, config['epochs']):
        loss_val = train(epoch, config, data, model, optimizer)

        if(loss_val < best_loss):
            best_loss = loss_val
            bad_counter = 0
        else:
            bad_counter += 1
        # print('bad_counter: {}'.format(bad_counter))
        if(bad_counter == config['patience']):
            break
    test_acc = test(config, data, model)

    return test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', type=str, default='JKNet_CiteSeer')
    args = parser.parse_args()

    with open('./config.yaml') as file:
        obj = yaml.safe_load(file)
        config = obj[args.key]

    test_acc = np.zeros(config['n_tri'])
    for tri in range(config['n_tri']):
        test_acc[tri] = run(config)
    print('config: {}'.format(config))
    print('\twhole test accuracies({} tries) = {}'.format(config['n_tri'], test_acc))
    print('\tave: {:.3f} max: {:.3f} min: {:.3f}' \
            .format(np.mean(test_acc), np.max(test_acc), np.min(test_acc)))
    

if __name__ == "__main__":
    main()
