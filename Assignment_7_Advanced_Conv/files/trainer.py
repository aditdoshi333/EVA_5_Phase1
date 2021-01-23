import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm.auto import tqdm, trange


class Training:

    def __init__(self, model, optimizer, train_loader, test_loader,device ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion =  nn.CrossEntropyLoss()


    def train(self, n_epoch=15):

        train_losses = []
        test_losses = []
        train_acc = []
        test_acc = []

        print('Training the model for {} epochs'.format(n_epoch))

        for epoch in range(1, n_epoch + 1):

            print('EPOCH:', epoch)

            # train the model
            current_train_acc, current_train_loss = self.train_every_epoch(epoch)

            # validate the model
            current_test_acc, current_train_acc = self.test()

            train_acc.extend(current_train_acc)
            train_losses.extend(current_train_loss)
            test_acc.extend(current_test_acc)
            test_losses.extend(current_train_acc)

        return (train_acc, train_losses), (test_acc, test_losses)

    def train_every_epoch(self, epoch):

        train_acc = []
        train_loss = []
        correct = 0
        processed = 0

        self.model.train()

        pbar = tqdm(self.train_loader, ncols="80%")

        for batch_idx, (data, target) in enumerate(pbar):

            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            # Predict
            y_pred = self.model(data)

            # Calculate loss

            loss = self.criterion(y_pred, target)

            train_loss.append(loss.data.cpu().numpy().item())

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            # get the index of the max log-probability
            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            # Update pbar-tqdm
            pbar.set_description(
                desc=f'epoch={epoch} loss={loss.item()} batch_id={batch_idx} accuracy={100 * correct / processed:0.2f}')
            train_acc.append(100 * correct / processed)

        return (train_acc, train_loss)

    def test(self):
        test_losses = []
        test_acc = []

        # set the model in evaluation mode
        self.model.eval()

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                # sum up batch loss
                test_loss += self.criterion(output, target).item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

        test_acc.append(100. * correct / len(self.test_loader.dataset))

        return test_acc, test_losses

    def get_misclassified(self):
        misclassified = []
        misclassified_pred = []
        misclassified_target = []
        # put the model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for data, target in self.test_loader:
                # move them to the respective device
                data, target = data.to(self.device), target.to(self.device)
                # do inferencing
                output = self.model(data)
                # get the predicted output
                pred = output.argmax(dim=1, keepdim=True)

                # get the current misclassified in this batch
                list_misclassified = (pred.eq(target.view_as(pred)) == False)
                batch_misclassified = data[list_misclassified]
                batch_mis_pred = pred[list_misclassified]
                batch_mis_target = target.view_as(pred)[list_misclassified]

                misclassified.append(batch_misclassified)
                misclassified_pred.append(batch_mis_pred)
                misclassified_target.append(batch_mis_target)

        # group all the batched together
        misclassified = torch.cat(misclassified)
        misclassified_pred = torch.cat(misclassified_pred)
        misclassified_target = torch.cat(misclassified_target)

        return list(map(lambda x, y, z: (x, y, z), misclassified, misclassified_pred, misclassified_target))