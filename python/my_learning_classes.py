import torch
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.utils as U
from torch_geometric.nn import GCNConv, NNConv
from torch_scatter import scatter_max
from torch_geometric.data import Data, Batch, DataLoader
import random
import os


node_features_num = 4
abstract_feature_num = 32
edge_features = 16
MLP_hidden_length = 128

class twoLayerNet1(torch.nn.Module):
    def __init__(self):
        super(twoLayerNet1, self).__init__()
        self.fully_1 = torch.nn.Linear(edge_features, MLP_hidden_length)
        self.dense1_bn = torch.nn.BatchNorm1d(MLP_hidden_length)
        self.fully_2 = torch.nn.Linear(MLP_hidden_length, node_features_num * (abstract_feature_num - 4))
    def forward(self, x):
        x = self.fully_1(x)
        x = self.dense1_bn(x)
        x = F.relu(x)
        x = self.fully_2(x)
        x = F.relu(x)
        return x

class twoLayerNet2(torch.nn.Module):
    def __init__(self):
        super(twoLayerNet2, self).__init__()
        self.fully_1 = torch.nn.Linear(edge_features, MLP_hidden_length)
        self.dense1_bn = torch.nn.BatchNorm1d(MLP_hidden_length)
        self.fully_2 = torch.nn.Linear(MLP_hidden_length, abstract_feature_num * abstract_feature_num)

    def forward(self, x):
        x = self.fully_1(x)
        x = self.dense1_bn(x)
        x = F.relu(x)
        x = self.fully_2(x)
        x = F.relu(x)
        return x

class twoLayerNet3(torch.nn.Module):
    def __init__(self):
        super(twoLayerNet3, self).__init__()
        self.fully_1 = torch.nn.Linear(edge_features, MLP_hidden_length)
        self.dense1_bn = torch.nn.BatchNorm1d(MLP_hidden_length)
        self.fully_2 = torch.nn.Linear(MLP_hidden_length, abstract_feature_num * abstract_feature_num)

    def forward(self, x):
        x = self.fully_1(x)
        x = self.dense1_bn(x)
        x = F.relu(x)
        x = self.fully_2(x)
        x = F.relu(x)
        return x

class twoLayerNet4(torch.nn.Module):
    def __init__(self):
        super(twoLayerNet4, self).__init__()
        self.fully_1 = torch.nn.Linear(edge_features, MLP_hidden_length)
        self.dense1_bn = torch.nn.BatchNorm1d(MLP_hidden_length)
        self.fully_2 = torch.nn.Linear(MLP_hidden_length, abstract_feature_num * abstract_feature_num)

    def forward(self, x):
        x = self.fully_1(x)
        x = self.dense1_bn(x)
        x = F.relu(x)
        x = self.fully_2(x)
        x = F.relu(x)
        return x

class twoLayerNet5(torch.nn.Module):
    def __init__(self):
        super(twoLayerNet5, self).__init__()
        self.fully_1 = torch.nn.Linear(edge_features, MLP_hidden_length)
        self.dense1_bn = torch.nn.BatchNorm1d(MLP_hidden_length)
        self.fully_2 = torch.nn.Linear(MLP_hidden_length, abstract_feature_num * abstract_feature_num)

    def forward(self, x):
        x = self.fully_1(x)
        x = self.dense1_bn(x)
        x = F.relu(x)
        x = self.fully_2(x)
        x = F.relu(x)
        return x

class twoLayerNet6(torch.nn.Module):
    def __init__(self):
        super(twoLayerNet6, self).__init__()
        self.fully_1 = torch.nn.Linear(edge_features, MLP_hidden_length)
        self.dense1_bn = torch.nn.BatchNorm1d(MLP_hidden_length)
        self.fully_2 = torch.nn.Linear(MLP_hidden_length, abstract_feature_num * abstract_feature_num)

    def forward(self, x):
        x = self.fully_1(x)
        x = self.dense1_bn(x)
        x = F.relu(x)
        x = self.fully_2(x)
        x = F.relu(x)
        return x

class twoLayerNet7(torch.nn.Module):
    def __init__(self):
        super(twoLayerNet7, self).__init__()
        self.fully_1 = torch.nn.Linear(edge_features, MLP_hidden_length)
        self.dense1_bn = torch.nn.BatchNorm1d(MLP_hidden_length)
        self.fully_2 = torch.nn.Linear(MLP_hidden_length, abstract_feature_num * abstract_feature_num)

    def forward(self, x):
        x = self.fully_1(x)
        x = self.dense1_bn(x)
        x = F.relu(x)
        x = self.fully_2(x)
        x = F.relu(x)
        return x

class twoLayerNet8(torch.nn.Module):
    def __init__(self):
        super(twoLayerNet8, self).__init__()
        self.fully_1 = torch.nn.Linear(edge_features, MLP_hidden_length)
        self.dense1_bn = torch.nn.BatchNorm1d(MLP_hidden_length)
        self.fully_2 = torch.nn.Linear(MLP_hidden_length, abstract_feature_num * abstract_feature_num)

    def forward(self, x):
        x = self.fully_1(x)
        x = self.dense1_bn(x)
        x = F.relu(x)
        x = self.fully_2(x)
        x = F.relu(x)
        return x

class twoLayerNet9(torch.nn.Module):
    def __init__(self):
        super(twoLayerNet9, self).__init__()
        self.fully_1 = torch.nn.Linear(edge_features, MLP_hidden_length)
        self.dense1_bn = torch.nn.BatchNorm1d(MLP_hidden_length)
        self.fully_2 = torch.nn.Linear(MLP_hidden_length, abstract_feature_num * abstract_feature_num)

    def forward(self, x):
        x = self.fully_1(x)
        x = self.dense1_bn(x)
        x = F.relu(x)
        x = self.fully_2(x)
        x = F.relu(x)
        return x

class twoLayerNet10(torch.nn.Module):
    def __init__(self):
        super(twoLayerNet10, self).__init__()
        self.fully_1 = torch.nn.Linear(edge_features, MLP_hidden_length)
        self.dense1_bn = torch.nn.BatchNorm1d(MLP_hidden_length)
        self.fully_2 = torch.nn.Linear(MLP_hidden_length, abstract_feature_num * 1)

    def forward(self, x):
        x = self.fully_1(x)
        x = self.dense1_bn(x)
        x = F.relu(x)
        x = self.fully_2(x)
        x = F.relu(x)
        return x



class NetVariableClasses(torch.nn.Module):
    def __init__(self):
        super(NetVariableClasses, self).__init__()
        self.conv1 = NNConv(node_features_num, abstract_feature_num - 4, twoLayerNet1())
        self.conv2 = NNConv(abstract_feature_num, abstract_feature_num, twoLayerNet2())
        self.conv3 = NNConv(abstract_feature_num, abstract_feature_num, twoLayerNet3())
        self.conv4 = NNConv(abstract_feature_num, abstract_feature_num, twoLayerNet4())
        self.conv5 = NNConv(abstract_feature_num, abstract_feature_num, twoLayerNet5())
        self.conv6 = NNConv(abstract_feature_num, abstract_feature_num, twoLayerNet6())
        self.conv7 = NNConv(abstract_feature_num, abstract_feature_num, twoLayerNet7())
        self.conv8 = NNConv(abstract_feature_num, abstract_feature_num, twoLayerNet8())
        self.conv9 = NNConv(abstract_feature_num, abstract_feature_num, twoLayerNet9())
        self.conv10 = NNConv(abstract_feature_num, 1, twoLayerNet10())

        self.dense1_bn = torch.nn.BatchNorm1d(abstract_feature_num - 4)
        self.dense2_bn = torch.nn.BatchNorm1d(abstract_feature_num)
        self.dense3_bn = torch.nn.BatchNorm1d(abstract_feature_num)
        self.dense4_bn = torch.nn.BatchNorm1d(abstract_feature_num)
        self.dense5_bn = torch.nn.BatchNorm1d(abstract_feature_num)
        self.dense6_bn = torch.nn.BatchNorm1d(abstract_feature_num)
        self.dense7_bn = torch.nn.BatchNorm1d(abstract_feature_num)
        self.dense8_bn = torch.nn.BatchNorm1d(abstract_feature_num)
        self.dense9_bn = torch.nn.BatchNorm1d(abstract_feature_num)
        self.drop1 = torch.nn.Dropout(p=0.1)


    def forward(self, data):

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x_skip = x
        x = self.conv1(x, edge_index, edge_attr)
        x = self.dense1_bn(x)
        x = F.relu(torch.cat((x, x_skip), dim = 1))
        x_skip = x
        '''
        x = self.conv2(x, edge_index, edge_attr)
        x = self.dense2_bn(x)
        x = F.relu(x + x_skip)
        x_skip = x
        x = self.conv3(x, edge_index, edge_attr)
        x = self.dense3_bn(x)
        x = F.relu(x + x_skip)
        x_skip = x
        x = self.conv4(x, edge_index, edge_attr)
        x = self.dense4_bn(x)
        x = F.relu(x + x_skip)
        x_skip = x
        x = self.conv5(x, edge_index, edge_attr)
        x = self.dense5_bn(x)
        x = F.relu(x + x_skip)
        x_skip = x
        x = self.conv6(x, edge_index, edge_attr)
        x = self.dense6_bn(x)
        x = F.relu(x + x_skip)
        x_skip = x
        '''
        x = self.conv7(x, edge_index, edge_attr)
        x = self.dense7_bn(x)
        x = F.relu(x + x_skip)
        x_skip = x
        x = self.conv8(x, edge_index, edge_attr)
        x = self.dense8_bn(x)
        x = F.relu(x + x_skip)
        x_skip = x
        x = self.conv9(x, edge_index, edge_attr)
        x = self.dense9_bn(x)
        x = F.relu(x + x_skip)
        x = self.drop1(x)
        x = self.conv10(x, edge_index, edge_attr)

        return torch.clamp(F.sigmoid(x), min=0.000001, max=0.999999), x

#Numpy matrix (array) as input to COO format, with edge weights
def matrixToCOO_with_weights(adjacency, edge_weighs):
    s = adjacency.shape
    adjacency_pairs = []
    edge_attr = []

    for i in range(s[0]):

        if adjacency[i,i] == 1:
            adjacency_pairs.append(np.array([i,i]))
            edge_attr.append(np.array([edge_weighs[i,i]]))

        for j in range(i+1,s[1]):

            if adjacency[i,j] == 1:
                adjacency_pairs.append(np.array([i,j]))
                edge_attr.append(np.array([edge_weighs[i,j]]))
            if adjacency[j,i] == 1:
                adjacency_pairs.append(np.array([j,i]))
                edge_attr.append(np.array([edge_weighs[j,i]]))

    adjacency_pairs = torch.from_numpy(np.array(adjacency_pairs)).t().contiguous()
    edge_attr = torch.from_numpy(np.array(edge_attr)).float()

    return adjacency_pairs, edge_attr

#Numpy matrix (array) as input to COO format, without edge weights
def matrixToCOO(adjacency):
    s = adjacency.shape
    adjacency_pairs = []

    for i in range(s[0]):

        if adjacency[i,i] == 1:
            adjacency_pairs.append(np.array([i,i]))

        for j in range(i+1,s[1]):

            if adjacency[i,j] == 1:
                adjacency_pairs.append(np.array([i,j]))
            if adjacency[j,i] == 1:
                adjacency_pairs.append(np.array([j,i]))

    adjacency_pairs = torch.from_numpy(np.array(adjacency_pairs)).t().contiguous()

    return adjacency_pairs

def one_hot(seq,depth,device):
    ones = torch.sparse.torch.eye(depth)
    ones = ones.to(device)
    return ones.index_select(0,seq)


#Softmax loss
def diff_node_loss(x_hat, y, indices, device, num_true_nodes):
    num_graphs = y.size()[0]
    chunk_size = []
    ct = 0

    for i in range(indices.size()[0]-1):
        if indices[i] != indices[i+1]:
            chunk_size.append(i - ct + 1)
            ct = i + 1

    chunk_size.append(indices.size()[0] - ct)
    list_of_chunks = torch.split(x_hat, chunk_size, dim=0)
    i = 0
    mean_loss = torch.tensor(0.).to(device)

    for chunk in list_of_chunks:
        label_size = num_true_nodes[i]
        chunk = torch.t(chunk)
        chunk = F.log_softmax(chunk[:,0:num_true_nodes[i]], dim=1)
        curr_label = one_hot(y[i], label_size, device)

        mean_loss = mean_loss + torch.sum(-chunk * curr_label, dim=1)
        i = i + 1

    mean_loss = mean_loss / num_graphs

    return mean_loss

#Sigmoid loss
def diff_node_loss_bis(x_hat, y, indices, device, num_true_nodes, activatedIndices):

    chunk_size = []
    ct = 0

    for i in range(indices.size()[0]-1):
        if indices[i] != indices[i+1]:
            chunk_size.append(i - ct + 1)
            ct = i + 1

    chunk_size.append(indices.size()[0] - ct)
    list_of_chunks = torch.split(x_hat, chunk_size, dim=0)
    i = 0
    mean_loss = torch.tensor(0.).to(device)

    current_tr_ind = 0
    for chunk in list_of_chunks:
        temp_y = y[current_tr_ind: current_tr_ind + num_true_nodes[i]]
        temp_activated = activatedIndices[current_tr_ind: current_tr_ind + num_true_nodes[i]].clone()
        current_tr_ind = current_tr_ind + num_true_nodes[i]

        #weighting for wait action
        temp_activated[-1] = temp_activated[-1] * (0.3 * (temp_activated == 1).sum())


        chunk = torch.t(chunk)[0]
        chunk = torch.clamp(F.sigmoid(chunk[0:num_true_nodes[i]]), min=0.01, max=0.99)
        curr_label = temp_y
        error = torch.sum(- temp_activated * curr_label * torch.log(chunk) - temp_activated * (1 - curr_label) * torch.log(1 - chunk))

        mean_loss = mean_loss + error
        i = i + 1

    mean_loss = mean_loss / i

    return mean_loss

#Accuracy for softmax
def diff_node_acc(x_hat, y, indices, device):
    num_graphs = y.size()[0]
    chunk_size = []
    ct = 0

    for i in range(indices.size()[0]-1):
        if indices[i] != indices[i+1]:
            chunk_size.append(i - ct + 1)
            ct = i + 1

    chunk_size.append(indices.size()[0] - ct)
    list_of_chunks = torch.split(x_hat, chunk_size, dim=0)
    i = 0
    acc = []

    for chunk in list_of_chunks:
        label_size = chunk.size()[0]
        chunk = torch.t(chunk)
        chunk = F.log_softmax(chunk, dim=1)
        curr_label = one_hot(y[i], label_size, device)
        #print((torch.argmax(chunk, dim=1)))
        #print(torch.tensor([y[i]]))
        acc.append((torch.argmax(chunk, dim=1) == torch.tensor([y[i]]).to(device)).float())
        i = i + 1

    #print(acc)
    return torch.mean(torch.cat(acc))

def diff_node_acc_bis(x_hat, y, indices, num_graphs, device):

    sig_x_hat = torch.clamp(F.sigmoid(x_hat), min=0.000001, max=0.999999)
    shaped_y = y.reshape(y.size()[0],1)
    pred = (sig_x_hat > 0.5).float()
    acc = (pred == shaped_y).float()
    accuracy = torch.squeeze(torch.mean(acc, dim=0))
    return accuracy

def diff_node_acc_nodes(x_hat, y, indices, device, num_true_nodes, activatedIndices):

    chunk_size = []
    ct = 0
    acc = []
    num_nodes_on = 0

    for i in range(indices.size()[0]-1):
        if indices[i] != indices[i+1]:
            chunk_size.append(i - ct + 1)
            ct = i + 1

    chunk_size.append(indices.size()[0] - ct)
    list_of_chunks = torch.split(x_hat, chunk_size, dim=0)
    i = 0

    current_tr_ind = 0
    for chunk in list_of_chunks:
        temp_y = y[current_tr_ind: current_tr_ind + num_true_nodes[i]]
        temp_activated = activatedIndices[current_tr_ind: current_tr_ind + num_true_nodes[i]]
        current_tr_ind = current_tr_ind + num_true_nodes[i]

        chunk = torch.t(chunk)[0]
        chunk = torch.clamp(F.sigmoid(chunk[0:num_true_nodes[i]]), min=0.001, max=0.999)
        curr_label = temp_y
        for index in range(len(curr_label)):
            if temp_activated[index] == 1:
                if (chunk[index] > 0.5) == (curr_label[index]):
                    acc.append(1)
                else:
                    acc.append(0)

        i = i + 1

    return acc

def soft_random_train(model, window, steps, batch_size, optimizer, device):
    if batch_size > len(window):
        batch_size = len(window)
    if len(window) > 0:
        model.train()
        for i in range(steps):
            mini_batch_list = random.sample(window, batch_size)
            mini_batch_matrix = Batch.from_data_list(mini_batch_list)
            mini_batch_matrix = mini_batch_matrix.to(device)
            sig_x, x_hat = model(mini_batch_matrix)
            loss = diff_node_loss_bis(x_hat, mini_batch_matrix.y, mini_batch_matrix.batch, device,
                                      mini_batch_matrix.num_true_nodes, mini_batch_matrix.activatedIndices)
            #print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(loss)
        model.eval()
    return

def split_and_train(model, window, steps, batch_size, optimizer, device):
    test = window[-100:]
    window = window[0:-100]
    if batch_size > len(window):
        batch_size = len(window)
    if len(window) > 0:
        model.train()
        k = 0
        for i in range(steps):
            ind = 0
            while ind < len(window):
                if ind + batch_size < len(window):
                    mini_batch_list = window[ind : (ind + batch_size)]
                else:
                    mini_batch_list = window[ind:]
                mini_batch_matrix = Batch.from_data_list(mini_batch_list)
                mini_batch_matrix = mini_batch_matrix.to(device)
                sig_x, x_hat = model(mini_batch_matrix)
                loss = diff_node_loss_bis(x_hat, mini_batch_matrix.y, mini_batch_matrix.batch, device,
                                          mini_batch_matrix.num_true_nodes, mini_batch_matrix.activatedIndices)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("ind = %d l = %f"%(ind, loss))
                ind = ind + batch_size

                if k % 20 == 0:
                    model.eval()
                    batch_matrix = Batch.from_data_list(test)
                    batch_matrix = batch_matrix.to(device)
                    sig_x, x_hat = model(batch_matrix)
                    loss = diff_node_loss_bis(x_hat, batch_matrix.y, batch_matrix.batch, device,
                                              batch_matrix.num_true_nodes, batch_matrix.activatedIndices)
                    print("test loss = %f" % loss)
                    model.train()
                k = k + 1

    model.eval()
    return

def split_and_train_parts(model, steps, batch_size, optimizer, device):

    dir = "x/"
    directory = os.fsencode("x/")
    test = torch.load("x/validation/data0.pt")
    test = test + torch.load("x/validation/data1.pt")
    test = test + torch.load("x/validation/data2.pt")
    test = test + torch.load("x/validation/data1000.pt")
    test = test + torch.load("x/validation/data1001.pt")

    min = 10000000
    k = 0
    for ep in range(steps):
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".pt"):
                window = torch.load("x/" + str(filename))

                if len(window) > 0:
                    model.train()
                    ind = 0
                    while ind < len(window):
                        if ind + batch_size < len(window):
                            mini_batch_list = window[ind : (ind + batch_size)]
                        else:
                            mini_batch_list = window[ind:]
                        mini_batch_matrix = Batch.from_data_list(mini_batch_list)
                        mini_batch_matrix = mini_batch_matrix.to(device)
                        sig_x, x_hat = model(mini_batch_matrix)
                        loss = diff_node_loss_bis(x_hat, mini_batch_matrix.y, mini_batch_matrix.batch, device,
                                                  mini_batch_matrix.num_true_nodes, mini_batch_matrix.activatedIndices)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        print("ep = %d, win = %s, ind = %d l = %f"%(ep, str(filename), ind, loss.item()))
                        ind = ind + batch_size

                        if k % 2000 == 0:
                            train_acc = diff_node_acc_nodes(x_hat, mini_batch_matrix.y, mini_batch_matrix.batch, device,
                                                      mini_batch_matrix.num_true_nodes,
                                                      mini_batch_matrix.activatedIndices)
                            print("ep = %d, win = %s, ind = %d train acc = %f" % (ep, str(filename), ind, np.mean(train_acc)))

                            model.eval()
                            loss_tab = []
                            pointer = 0
                            processed_ex = 0
                            test_acc = []
                            while pointer < len(test):
                                if pointer + batch_size < len(test):
                                    batch_list = test[pointer: (pointer + batch_size)]
                                else:
                                    batch_list = test[pointer:]

                                size = len(batch_list)
                                batch_matrix = Batch.from_data_list(batch_list)
                                batch_matrix = batch_matrix.to(device)
                                sig_x, x_hat = model(batch_matrix)
                                loss = diff_node_loss_bis(x_hat, batch_matrix.y, batch_matrix.batch, device,
                                                          batch_matrix.num_true_nodes, batch_matrix.activatedIndices) * size
                                loss_tab.append(loss.item())
                                list_acc = diff_node_acc_nodes(x_hat, batch_matrix.y, batch_matrix.batch, device,
                                                          batch_matrix.num_true_nodes,
                                                          batch_matrix.activatedIndices)
                                test_acc = test_acc + list_acc
                                pointer = pointer + batch_size

                            loss_tab = np.array(loss_tab)
                            avg_test_loss = np.sum(loss_tab)/len(test)
                            print("test loss = %f" % avg_test_loss)
                            print("test acc = %f" % np.mean(test_acc))

                            if avg_test_loss < min:
                                torch.save(model.state_dict(), "save_single_2-6-out")
                                min = avg_test_loss

                            model.train()
                        k = k + 1

    model.eval()
    return