import sys
sys.path.append('C:/Users/iratx/Documents/PROYECTOS/ml-project/ml_project/')
import pretty_errors
from datetime import datetime

pretty_errors.config.display_timestamp = True
pretty_errors.config.timestamp_function = lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S')

from utils.plots import plot
from model import BertClassifier

import torch
from torch import nn
from torch.optim import Adam
from dvc.api import params_show
from tqdm import tqdm
import pickle as pkl
from icecream import ic as print
print.configureOutput(prefix=f'{datetime.now()}|> ')

def train(model, train, test, learning_rate, epochs, batch_size):
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    best_acc = 0
    all_train_acc, all_train_loss, all_test_acc, all_test_loss = [], [], [], []

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()
            
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        
        total_acc_test = 0
        total_loss_test = 0

        with torch.no_grad():

            for test_input, test_label in test_dataloader:

                test_label = test_label.to(device)
                mask = test_input['attention_mask'].to(device)
                input_id = test_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, test_label.long())
                total_loss_test += batch_loss.item()
                
                acc = (output.argmax(dim=1) == test_label).sum().item()
                total_acc_test += acc
        
        print(f'\nEpochs: {epoch_num + 1}\nTrain Loss: {total_loss_train / len(train): .3f}\nTrain Accuracy: {total_acc_train / len(train): .3f}\nTest Loss: {total_loss_test / len(test): .3f}\nTest Accuracy: {total_acc_test / len(test): .3f}')
        
        if total_acc_test / len(test) > best_acc:
            torch.save(model, 'models/model.pth')
            best_acc = total_acc_test / len(test)

        all_test_acc.append(total_acc_test / len(test))
        all_test_loss.append(total_loss_test / len(test))
        all_train_acc.append(total_acc_train / len(train))
        all_train_loss.append(total_loss_train / len(train))

    return all_test_acc, all_test_loss, all_train_acc, all_train_loss

if __name__ == '__main__':
    params = params_show()["train"]               

    model = BertClassifier(dropout=params['dropout'])
    with open('data/datasets.pkl', 'rb') as f:
        sys.path.append('C:/Users/iratx/Documents/PROYECTOS/ml-project/ml_project/preprocessing')
        datasets = pkl.load(f)
        train_ds, test_ds = datasets['train'], datasets['test']

    test_acc, test_loss, train_acc, train_loss = train(model, train_ds, test_ds, params['lr'], params['epochs'], params['batch_size'])
    plot(train_loss, train_acc, test_loss, test_acc)
