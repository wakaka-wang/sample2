import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset, DataLoader
import os
import copy

# Hyper parameters
num_epochs = 1
num_classes = 131
batch_size = 24
learning_rate = 0.001

# import data and image processing
data_transforms = {
    'Training': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]),
    'Test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]),
}

data_dir = 'Dataset'  # file path of the data
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])  # ImageFolder 数据加载器
                  for x in ['Training', 'Test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                              shuffle=True, num_workers=0)
               for x in ['Training', 'Test']}
dataset_sizes = {x: len(image_datasets[x])
                 for x in ['Training', 'Test']}
class_names = image_datasets['Training'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('import data and image processing')


# visualize a few images
def imshow(image, title=None):  # change inp to image
    """Imshow for Tensor."""
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.1)  # pause is a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['Training']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
print(len(os.listdir("Dataset/Training")))  # number of classes


# build CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 参数的定义
        # 卷积层+池化层
        self.conv = nn.Sequential(
            # 第一层
            # 卷积核大小#*3，包含32个卷积核，relu激活，same padding形式
            # 第一池化层，池化大小2*2，步长1，valid padding形式
            nn.Conv2d(kernel_size=3, in_channels=3, out_channels=32, padding=1),  # (64,64,32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),  # 63,63,32

            # 第二层
            # 卷积核大小3*3，包含64个卷积核，relu激活，same padding形式
            # 第二池化层，池化大小2*2，步长1，valid padding
            nn.Conv2d(kernel_size=3, in_channels=32, out_channels=64, padding=1),  # 63, 63, 64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=1),  # 62, 62, 64

            # 第三层
            # 卷积核大小3*3，包含128个卷积核，relu激活，same padding形式
            # 第三层池化，池化大小2*2，步长2，valid padding
            nn.Conv2d(kernel_size=3, in_channels=64, out_channels=128, padding=1),  # 62,62,128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)  # 31,31,128
        )
        # fully connected layer
        self.line = nn.Sequential(
            nn.Linear(in_features=10580, out_features=128), # 31 31 128
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=5)
        )

        # 以下是旧的代码
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(10580, 1024)
        # self.fc2 = nn.Linear(1024, 5)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.line(x)
        return x

    # 以下是旧代码 搭配旧网络结构
    # def forward(self, x):
    #     x = F.relu(F.max_pool2d(self.conv1(x), 2))
    #     x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    #     x = x.view(x.shape[0], -1)
    #     x = F.relu(self.fc1(x))
    #     x = F.dropout(x, training=self.training)
    #     x = self.fc2(x)
    #
    #     return x


model = CNN()
print(model)


# training the model
def train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs):
    # global inputs
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # each epoch has a training and validation phase
        for phase in ['Training', 'Test']:
            if phase == 'Training':
                model.train()  # set model to training phase
            else:
                model.eval()  # set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Training'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)  # ###
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'Training':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'Training':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'Test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Visualizing the model predictions
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['Test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


# optimization
model_ft = CNN().to(device)
criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Decay LR by a factor of 0.1 every 7 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

# Train and evaluate
model_ft = train_model(model_ft, criterion, optimizer, scheduler, num_epochs=num_epochs)

visualize_model(model_ft)
print('show predicts')

plt.ioff()
plt.show()
