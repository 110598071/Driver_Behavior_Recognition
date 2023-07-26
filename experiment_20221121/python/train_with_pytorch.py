import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import util
import feature_data
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAVE_MODEL = True

input_size = 38
hidden_size = (30, 20, 15)
num_classes = 10
num_epochs = 15
batch_size = 150
learning_rate = 0.001
K_FOLD = 10

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0]) 
        self.bn1 = nn.BatchNorm1d(hidden_size[0])
        self.hiddenActivation1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1]) 
        self.bn2 = nn.BatchNorm1d(hidden_size[1])
        self.hiddenActivation2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.bn3 = nn.BatchNorm1d(hidden_size[2])
        self.hiddenActivation3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size[2], num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.hiddenActivation1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.hiddenActivation2(out)
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.hiddenActivation3(out)
        out = self.fc4(out)
        return out

def get_batch_data_and_label(i, data_amount, data_loader, index_list):
    batch_data = []
    batch_label = []
    for j in range(batch_size):
        index = i*batch_size + j
        if index >= data_amount: break
        batch_data.append(data_loader[0][index_list[index]])
        batch_label.append(data_loader[1][index_list[index]])
    return batch_data, batch_label

def train(model, criterion, optimizer, train_loader, isCrossValidation=False):
    loss_list = []
    train_data_amount = len(train_loader[0])
    train_iterator = int(np.ceil(train_data_amount / batch_size))
    index_list = [i for i in range(train_data_amount)]
    for epoch in range(num_epochs):
        np.random.shuffle(index_list)
        for i in range(train_iterator):
            batch_data, batch_label = get_batch_data_and_label(i, train_data_amount, train_loader, index_list)

            images = torch.tensor(np.asarray(batch_data)).float()
            labels = torch.tensor(np.asarray(batch_label)).long()

            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_list.append(loss.item())
            if (not isCrossValidation):
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, train_data_amount if (i+1)*batch_size > train_data_amount else (i+1)*batch_size, train_data_amount, loss.item()))
    return model, loss_list

def test(model, test_loader, matrix_list = [[0]*10 for _ in range(10)]):
    test_data_amount = len(test_loader[0])
    test_iterator = int(np.ceil(test_data_amount / batch_size))
    
    with torch.no_grad():
        correct = 0
        for i in range(test_iterator):
            batch_data, batch_label = get_batch_data_and_label(i, test_data_amount, test_loader, [i for i in range(test_data_amount)])

            images = torch.tensor(np.asarray(batch_data)).float()
            labels = torch.tensor(np.asarray(batch_label)).long()

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            for j, output in enumerate(outputs):
                label = labels[j]
                _, predicted = torch.max(output.data, 0)
                if (predicted == label):
                    correct += 1
                matrix_list[label][predicted] += 1
        final_accuray = np.around(100 * correct / test_data_amount, 2)
        print('Accuracy of the network on the test images: {} %'.format(final_accuray))
    return matrix_list, final_accuray

def save_model(model, final_accuray):
    FILE_LASTNAME = get_file_last_name(final_accuray)
    MODEL_FILEPATH = '../model/pytorch_neural_network/model' + FILE_LASTNAME + '.pth'
    torch.save(model.state_dict(), MODEL_FILEPATH)

def draw_confusion_matrix(matrix_list, final_accuray, isCrossValidation=False):
    FILE_LASTNAME = get_file_last_name(final_accuray)
    if (isCrossValidation):
        FILE_LASTNAME = "_cross_validation" + FILE_LASTNAME

    HEATMAP_PATH = '../heatmap/pytorch_neural_network/nn_heatmap' + FILE_LASTNAME + '.png'
    for i, row in enumerate(matrix_list):
        row_sum = np.sum(row)
        for j, num in enumerate(row):
            matrix_list[i][j] = np.around(num/row_sum, 2)

    ticks = [list(range(10))]
    df = pd.DataFrame(matrix_list, columns=ticks, index=ticks)

    plt.figure()
    sns.heatmap(df, annot=True)
    plt.title('confusion_matrix')
    plt.ylabel('actual label')
    plt.xlabel('predicted label')
    plt.savefig(HEATMAP_PATH, dpi=300)

def draw_loss_plot(loss_list, final_accuray):
    FILE_LASTNAME = get_file_last_name(final_accuray)
    LOSSPLOT_PATH = '../lossplot/lossplot' + FILE_LASTNAME + '.png'

    ticks = [i for i in range(len(loss_list))]
    plt.figure()
    plt.title('loss plot')
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.xticks(ticks[::100])
    plt.plot(loss_list)
    plt.savefig(LOSSPLOT_PATH, dpi=300)

def get_file_last_name(final_accuray):
    time = datetime.now().strftime("_%Y%m%d_%H%M")
    FILE_LASTNAME = time + '_' + str(final_accuray) + '%_epoch' + str(num_epochs) + '_batch' + str(batch_size)
    return FILE_LASTNAME

def get_confusion_matrix_with_cross_validation():
    kf = KFold(n_splits = K_FOLD)
    action_data, action_label = util.get_np_asarray_data_and_label()
    matrix_list = [[0]*10 for _ in range(10)]
    accuracy_list = []

    for train_index, test_index in kf.split(action_data):
        train_loader = (action_data[train_index], action_label[train_index])
        test_loader = (action_data[test_index], action_label[test_index])

        model = NeuralNet(input_size, hidden_size, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        model, _ = train(model, criterion, optimizer, train_loader, True)
        matrix_list, final_accuray = test(model, test_loader, matrix_list)
        accuracy_list.append(final_accuray)

    print("==============================================")
    print('Accuracy of the network with cross validation: {} %'.format(np.around(np.mean(accuracy_list), 2)))
    draw_confusion_matrix(matrix_list, final_accuray, True)
    plt.show()

def train_nn_model():
    train_loader, test_loader = feature_data.get_train_and_test_data()

    # scaler = StandardScaler()
    # train_loader = (scaler.fit_transform(train_loader[0]), train_loader[1])
    # test_loader = (scaler.transform(test_loader[0]), test_loader[1])

    model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model, loss_list = train(model, criterion, optimizer, train_loader)
    matrix_list, final_accuray = test(model, test_loader)

    save_model(model, final_accuray)
    draw_confusion_matrix(matrix_list, final_accuray)
    draw_loss_plot(loss_list, final_accuray)
    plt.show()

if __name__ == '__main__':
    if SAVE_MODEL:
        train_nn_model()
    else:
        get_confusion_matrix_with_cross_validation()