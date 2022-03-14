import csv
from torchvision import datasets, transforms
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import math
import random
from itertools import permutations
import numpy as np
import time
from torch.utils.data import DataLoader, Dataset
import copy
from datetime import datetime
from models_fednn import *
from utils import *
torch.autograd.set_detect_anomaly(True)

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        d = int(self.idxs[item])
        image, label = self.dataset[d]
        return image, label

trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])


def noniid(dataset, num_users, shard_per_user, rand_set_all=[]):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(dataset.targets))
    shard_per_class = int(shard_per_user * num_users / num_classes)
    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(dataset.targets)[value])
        assert(len(x)) <= shard_per_user
        test.append(value)
    test = np.concatenate(test)
    assert(len(test) == len(dataset))
    assert(len(set(list(test))) == len(dataset))

    return dict_users, rand_set_all

def get_data(args):
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users_train = iid(dataset_train, args.num_users)
            dict_users_test = iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user)
            dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, rand_set_all=rand_set_all)
    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
        if args.iid:
            dict_users_train = iid(dataset_train, args.num_users)
            dict_users_test = iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user)
            dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, rand_set_all=rand_set_all)
    elif args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
        dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
        if args.iid:
            dict_users_train = iid(dataset_train, args.num_users)
            dict_users_test = iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user)
            dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, rand_set_all=rand_set_all)
    else:
        exit('Error: unrecognized dataset')

    return dataset_train, dataset_test, dict_users_train, dict_users_test

def get_model(args):
    if args.model == 'cnn' and args.dataset == 'cifar10':
        random_model = random.randint(2, 4)
        if(random_model==3):
            net_glob = CNNCifar2(args=args).to(args.device)
        elif(random_model==4):
            net_glob = CNNCifar3(args=args).to(args.device)
        elif(random_model==5):
            net_glob = CNNCifar4(args=args).to(args.device)
        else:
            net_glob = CNNCifar3(args=args).to(args.device)
        print(net_glob)
    elif args.model == 'cnn' and args.dataset == 'cifar100':
        random_model = random.randint(1, 2)
        if(random_model==1):
            net_glob = CNNCifar100_1(args=args).to(args.device)
        elif(random_model==2):
            net_glob = CNNCifar100_2(args=args).to(args.device)
        else:
            net_glob = CNNCifar100_1(args=args).to(args.device)
        print(net_glob)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
        print(net_glob)
    elif args.model == 'mlp' and args.dataset == 'mnist':
        random_model = random.randint(3, 5)
        if(random_model==3):
            net_glob = MLP3(dim_in=784, dim_hidden=256, dim_out=args.num_classes).to(args.device)
        elif(random_model==4):
            net_glob = MLP4(dim_in=784, dim_hidden=256, dim_out=args.num_classes).to(args.device)
        elif(random_model==5):
            net_glob = MLP4(dim_in=784, dim_hidden=256, dim_out=args.num_classes).to(args.device)
        print(net_glob)
    else:
        exit('Error: unrecognized model')
    

    return net_glob

def test_img_local_all(net, args, dataset_test, dict_users_test,w_locals=None,w_glob_keys=None, indd=None,dataset_train=None,dict_users_train=None, return_all=False):
    tot = 0
    num_idxxs = args.num_users
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    for idx in range(num_idxxs):
        net_local = copy.deepcopy(net[idx])
        if w_locals is not None:
            w_local = net_local.state_dict()
            for k in w_locals[idx].keys():
                w_local[k] = w_locals[idx][k]
            net_local.load_state_dict(w_local)
        net_local.eval()
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            a, b =  test_img_local(net_local, dataset_test, args,idx=dict_users_test[idx],indd=indd, user_idx=idx)
            tot += len(dataset_test[dict_users_test[idx]]['x'])
        else:
            a, b = test_img_local(net_local, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx]) 
            tot += len(dict_users_test[idx])
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            acc_test_local[idx] = a*len(dataset_test[dict_users_test[idx]]['x'])
            loss_test_local[idx] = b*len(dataset_test[dict_users_test[idx]]['x'])
        else:
            acc_test_local[idx] = a*len(dict_users_test[idx])
            loss_test_local[idx] = b*len(dict_users_test[idx])
        del net_local
    
    if return_all:
        return acc_test_local, loss_test_local
    return  sum(acc_test_local)/tot, sum(loss_test_local)/tot
  

def test_img_local(net_g, dataset, args,idx=None,indd=None, user_idx=-1, idxs=None):
    net_g.eval()
    test_loss = 0
    correct = 0

    # put LEAF data into proper format
    if 'femnist' in args.dataset:
        leaf=True
        datatest_new = []
        usr = idx
        for j in range(len(dataset[usr]['x'])):
            datatest_new.append((torch.reshape(torch.tensor(dataset[idx]['x'][j]),(1,28,28)),torch.tensor(dataset[idx]['y'][j])))
    elif 'sent140' in args.dataset:
        leaf=True
        datatest_new = []
        for j in range(len(dataset[idx]['x'])):
            datatest_new.append((dataset[idx]['x'][j],dataset[idx]['y'][j]))
    else:
        leaf=False
    
    if leaf:
        data_loader = DataLoader(DatasetSplit_leaf(datatest_new,np.ones(len(datatest_new))), batch_size=args.local_bs, shuffle=False)
    else:
        data_loader = DataLoader(DatasetSplit(dataset,idxs), batch_size=args.local_bs,shuffle=False)
    if 'sent140' in args.dataset:
        hidden_train = net_g.init_hidden(args.local_bs)
    count = 0
    for idx, (data, target) in enumerate(data_loader):
        if 'sent140' in args.dataset:
            input_data, target_data = process_x(data, indd), process_y(target, indd)
            if args.local_bs != 1 and input_data.shape[0] != args.local_bs:
                break

            data, targets = torch.from_numpy(input_data).to(args.device), torch.from_numpy(target_data).to(args.device)
            net_g.zero_grad()

            hidden_train = repackage_hidden(hidden_train)
            output, hidden_train = net_g(data, hidden_train)

            loss = F.cross_entropy(output.t(), torch.max(targets, 1)[1])
            _, pred_label = torch.max(output.t(), 1)
            correct += (pred_label == torch.max(targets, 1)[1]).sum().item()
            count += args.local_bs
            test_loss += loss.item()

        else:
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            log_probs, hidden = net_g(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    if 'sent140' not in args.dataset:
        count = len(data_loader.dataset)
    test_loss /= count
    accuracy = 100.00 * float(correct) / count
    return  accuracy, test_loss

class LinCKALoss(nn.Module):
    def __init__(self, weight=None, size_average=True, eta=None):
        super(LinCKALoss, self).__init__()
        self.eta = eta
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n])
        I = torch.eye(n)
        H = (I - unit / n).to(args.device)
        A = torch.matmul(H, K)
        b = torch.matmul(A, H)
        return b  


    def linear_HSIC(self, L_X, L_Y):
        #L_X = torch.matmul(X, X.T)
        #L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))


    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))
        return 1 - hsic / (var1 * var2)


    def forward(self, outputs, targets, glob_reps, local_reps, smooth=1):       
        ce = self.cross_entropy_loss(outputs, targets)

        if (self.eta!=0):
          err = self.linear_CKA(glob_reps, local_reps)
          #print(err,ce)
          final_err = self.eta*err + ce
        else :
          final_err = ce
        
        return final_err


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None, sample_images = None, eta = None):
        self.args = args
        self.loss_func = LinCKALoss(eta=eta)

        if 'femnist' in args.dataset or 'sent140' in args.dataset: 
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])),name=self.args.dataset), batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
         
        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        elif indd is not None:
            self.indd = indd
        else:
            self.indd=None        
        
        self.dataset=dataset
        self.idxs=idxs
        self.sample_images = sample_images


    def train(self, net, w_glob_keys, reps_global = None, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.1):
        bias_p=[]
        weight_p=[]
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
        [     
            {'params': weight_p, 'weight_decay':0.0001},
            {'params': bias_p, 'weight_decay':0}
        ],
        lr=lr, momentum=0.5
        )
        if self.args.alg == 'prox':
            optimizer = FedProx.FedProx(net.parameters(),
                             lr=lr,
                             gmf=self.args.gmf,
                             mu=self.args.mu,
                             ratio=1/self.args.num_users,
                             momentum=0.5,
                             nesterov = False,
                             weight_decay = 1e-4)
            
        local_eps = self.args.local_ep

        if last:
            w_glob_keys = []
            local_eps = 10
        
        head_eps = local_eps-self.args.local_rep_ep
        epoch_loss = []
        num_updates = 0

        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(self.args.local_bs)

        for iter in range(local_eps):
            done = False

            # for FedRep, first do local epochs for the head
            if (iter < head_eps and self.args.alg == 'fedrep') or last:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            
            # then do local epochs for the representation
            elif iter == head_eps and self.args.alg == 'fedrep' and not last:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            # all other methods update all parameters simultaneously
            elif self.args.alg != 'fedrep':
                for name, param in net.named_parameters():
                     param.requires_grad = True 
       
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                
                if 'sent140' in self.args.dataset:
                    input_data, target_data = process_x(images, self.indd), process_y(labels,self.indd)
                    if self.args.local_bs != 1 and input_data.shape[0] != self.args.local_bs:
                        break
                    net.train()
                    data, targets = torch.from_numpy(input_data).to(self.args.device), torch.from_numpy(target_data).to(self.args.device)
                    net.zero_grad()
                    hidden_train = repackage_hidden(hidden_train)
                    output, hidden_train = net(data, hidden_train)
                    loss = self.loss_func(output.t(), torch.max(targets, 1)[1])
                    loss.backward()
                    optimizer.step()
                else:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs,layerout = net(images)
                    sample_preds,layerout = net(self.sample_images.to(self.args.device))
                    #net.get_feat_vec(self.sample_images.to(self.args.device))
                    reps_local = torch.matmul(layerout,layerout.T)
                    loss = self.loss_func(log_probs, labels, reps_global, reps_local) 
                    #print(reps_global.shape)
                    #print(torch.norm(reps_global))                   
                    loss.backward(retain_graph=True)
                    optimizer.step()
                num_updates += 1
                batch_loss.append(loss.item())
                if num_updates == self.args.local_updates:
                    done = True
                    break
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            if done:
                break
            
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset" )
parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
parser.add_argument('--num_users', type=int, default=50, help="number of users: n")
parser.add_argument('--shard_per_user', type=int, default=5, help="classes per user")
parser.add_argument('--model', type=str, default='cnn', help="model name")
parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
parser.add_argument('--gpu', type=int, default=3, help="GPU ID, -1 for CPU")
parser.add_argument('--frac', type=float, default=0.2, help="the fraction of clients: C")
parser.add_argument('--alg', type=str, default='fedavg', help='FL algorithm to use')
parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
parser.add_argument('--m_tr', type=int, default=500, help="maximum number of samples/user to use for training")
parser.add_argument('--m_ft', type=int, default=500, help="maximum number of samples/user to use for fine-tuning")

parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
parser.add_argument('--bs', type=int, default=128, help="test batch size")
parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
parser.add_argument('--grad_norm', action='store_true', help='use_gradnorm_avging')
parser.add_argument('--lr_decay', type=float, default=1.0, help="learning rate decay per round")
parser.add_argument('--local_updates', type=int, default=1000000, help="maximum number of local updates")
parser.add_argument('--local_rep_ep', type=int, default=1, help="the number of local epochs for the representation for FedRep")
parser.add_argument('--test_freq', type=int, default=20, help='how often to test on val set')
parser.add_argument('--save_every', type=int, default=100, help='how often to save models')

args = parser.parse_args(args=[])

args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)

#net_glob = get_model(args)

#total_num_layers = len(net_glob.state_dict().keys())

#net_keys = [*net_glob.state_dict().keys()]

w_glob_keys = []

net_locals = {}
w_locals = {}

for user in range(args.num_users):
  w_local_dict = {}
  net_local_temp = get_model(args)
  for key in net_local_temp.state_dict().keys():
    w_local_dict[key] = net_local_temp.state_dict()[key]
  
  w_locals[user] = w_local_dict    
  net_locals[user] = net_local_temp

loss_train = []
accs = []
times = []
accs10 = 0
accs10_glob = 0
start = time.time()
lens = np.ones(args.num_users)
rep_size = 128
l = 1000

now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M")
run_comment = args.dataset + '_' + args.model + '_' + str(args.num_users) + 'users_' + str(args.shard_per_user) + '_classes_'
outfile = '/home/disha/Documents/fednn/fednn_scripts/results/hetero_lin_cka_' + run_comment + date_time +'.txt'
f = open(outfile, 'w')
writer = csv.writer(f)

test_loader = DataLoader(dataset_train, batch_size=l, shuffle=True)
for images, labels in test_loader:
    sample_images = images
    sample_labels = labels

for l in [1000]:
    print("l ----- ",str(l))
    for eta0 in [0.01, 0.01]:

        print("ETA0 ----- ",str(eta0))
        writer.writerow("Eta0 -------------------- " + str(eta0))

        global_reps=torch.zeros((l,l), device = args.device)
        
        net_local_list = []
                
        for iter in range(args.epochs+1):

                eta = eta0*math.sqrt(iter)
                print("Iter ",iter)
                w_glob = {}        
                loss_locals = []
                local_reps = {}

                m = max(int(args.frac * args.num_users), 1)
                if iter == args.epochs:
                    m = args.num_users

                idxs_users = np.random.choice(range(args.num_users), m, replace=False)
                w_keys_epoch = w_glob_keys
                times_in = []
                total_len=0
                
                for ind, idx in enumerate(idxs_users):
                    start_in = time.time()
                    if 'femnist' in args.dataset or 'sent140' in args.dataset:
                        if args.epochs == iter:
                            local = LocalUpdate(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_ft]], idxs=dict_users_train, indd=indd, sample_images = sample_images, eta=eta)
                        else:
                            local = LocalUpdate(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_tr]], idxs=dict_users_train, indd=indd, sample_images = sample_images, eta=eta)
                    else:
                        if args.epochs == iter:
                            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_ft], sample_images = sample_images, eta=eta)
                        else:
                            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_tr], sample_images = sample_images, eta=eta)

              
                    net_local = (net_locals[idx])
                    w_local = w_locals[idx]

                    for k in w_locals[idx].keys():
                        w_local[k] = w_locals[idx][k]

                    net_local.load_state_dict(w_local)

                    last = iter == args.epochs
                  
                    if 'femnist' in args.dataset or 'sent140' in args.dataset:
                        w_local, loss, indd = local.train(net=net_local.to(args.device), ind=idx, idx=clients[idx], w_glob_keys=w_glob_keys, reps_global = global_reps, lr=args.lr,last=last)
                    else:            
                        w_local, loss, indd = local.train(net=net_local.to(args.device), idx=idx, w_glob_keys=w_glob_keys, lr=args.lr, reps_global = global_reps, last=last)
                    loss_locals.append(copy.deepcopy(loss))


                    for k,key in enumerate(net_local.state_dict().keys()):
                        w_locals[idx][key] = w_local[key]


                    total_len += lens[idx]


                    #net_local_temp = copy.deepcopy(net_local) ## structure
                     ## newly learnt local model

                    #out, layerout = net_local_temp(sample_images.to(args.device))
                    #local_reps[idx] = layerout
                    times_in.append( time.time() - start_in )

                global_reps=torch.zeros((l,l),  device = args.device)
                
                local_reps = {}
                
                #print(l)
                test_loader = DataLoader(dataset_train, batch_size=l, shuffle=True, drop_last = True)

                for images, labels in test_loader:
                    sample_images = images

                print(sample_images.shape)
                for ind, idx in enumerate(idxs_users):
                  net_local_temp = copy.deepcopy(net_locals[idx])
                  net_local_temp.load_state_dict(w_locals[idx])
                  out, layerout = net_local_temp(sample_images.to(args.device))
                  local_reps[idx] = torch.matmul(layerout, layerout.T)
                  #print(global_reps.shape)
                  #print(local_reps[idx].shape)
                  global_reps = torch.add(global_reps,local_reps[idx])

                global_reps = torch.div(global_reps, m)
                loss_avg = sum(loss_locals) / len(loss_locals)
                loss_train.append(loss_avg)


                if iter % args.test_freq==args.test_freq-1 or iter>=args.epochs-10:
                    if times == []:
                        times.append(max(times_in))
                    else:
                        times.append(times[-1] + max(times_in))
                    acc_test, loss_test = test_img_local_all(net_locals, args, dataset_test, dict_users_test,
                                                                w_glob_keys=w_glob_keys, w_locals=w_locals,indd=indd,dataset_train=dataset_train, dict_users_train=dict_users_train, return_all=False)
                    accs.append(acc_test)
                    # for algs which learn a single global model, these are the local accuracies (computed using the locally updated versions of the global model at the end of each round)
                    if iter != args.epochs:
                        print('Round {:3d}, Eta: {:3f}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                                iter, eta, loss_avg, loss_test, acc_test))
                        writer.writerow('Round {:3d}, Eta: {:3f}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                                iter, eta, loss_avg, loss_test, acc_test))

                    else:
                        # in the final round, we sample all users, and for the algs which learn a single global model, we fine-tune the head for 10 local epochs for fair comparison with FedRep
                        print('Final Round, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                                loss_avg, loss_test, acc_test))
                        writer.writerow('Final Round, Eta: {:3f}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                                eta, loss_avg, loss_test, acc_test))

                    if iter >= args.epochs-10 and iter != args.epochs:
                        accs10 += acc_test/10

                    f.flush()
                
                    #if iter % args.save_every==args.save_every-1:
                    #    model_save_path = './models/hetero_lin_accs_'+ args.alg + '_' + args.dataset + '_' + str(args.num_users) +'_'+ str(args.shard_per_user) +'_iter' + str(iter)+ '.pt'
                    #    torch.save(net_glob.state_dict(), model_save_path)

                    
                    

        print('Average accuracy final 10 rounds: {}'.format(accs10))
        end = time.time()
        print(end-start)
        print(times)
        print(accs)

        writer.writerow('Average accuracy final 10 rounds: {}'.format(accs10))
        end = time.time()
        writer.writerow(str(end-start))
        writer.writerow(times)
        writer.writerow(accs)
