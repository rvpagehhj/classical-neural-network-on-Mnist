import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# 如果网络能在GPU中训练，就使用GPU；否则使用CPU进行训练
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 这个函数包括了两个操作：将图片转换为张量，以及将图片进行归一化处理
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Resize(64),
                                            torchvision.transforms.Normalize(mean=[0.5], std=[0.5])],
                                            )
path = './data/'  # 数据集下载后保存的目录

# 下载训练集和测试集
trainData = torchvision.datasets.MNIST(path, train=True, transform=transform, download=True)
testData = torchvision.datasets.MNIST(path, train=False, transform=transform)
# 设定每一个Batch的大小
BATCH_SIZE = 32
learning_rate = 0.0001
# 构建数据集和测试集的DataLoader
trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE)


# 定义t网络结构
class VGG16(nn.Module):
    def __init__(self, width_mult=1):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(  # 输入1*28*228
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear(2*2*512,1024)
        self.do1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.re =  nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 2*2*512)
        x = self.fc1(x)
        x = self.re(x)
        x = self.fc2(x)
        x = self.re(x)
        x = self.fc3(x)
        return x


net = VGG16().to(device)

summary(net, input_size=(1, 64, 64), batch_size=-1)

lossF = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
history = {'Test Loss': [], 'Test Accuracy': [], 'Loss': [], 'Acc': []}
# 打开网络的训练模式
net.train(True)
EPOCHS = 20  # 总的循环
for epoch in range(1, EPOCHS + 1):
    running_loss = 0
    running_acc = 0
    # 开始对训练集的DataLoader进行迭代
    for trainImgs, labels in trainDataLoader:
        # 将图像和标签传输进device中
        trainImgs = trainImgs.to(device)
        labels = labels.to(device)

        # 清空模型的梯度
        net.zero_grad()

        # 对模型进行前向推理
        outputs = net(trainImgs)

        # 计算本轮推理的Loss值
        loss = lossF(outputs, labels)
        running_loss += loss.item()
        # 计算本轮推理的准确率
        predictions = torch.argmax(outputs, dim=1)
        correct_num = torch.sum(predictions == labels)
        running_acc += correct_num.item()

        # 进行反向传播求出模型参数的梯度
        loss.backward()
        # 使用迭代器更新模型权重
        optimizer.step()

    running_loss /= len(trainDataLoader)
    running_acc /= len(trainData)
    # 将本step结果进行可视化处理
    print("[%d/%d] Train Loss: %.4f, Train Acc: %.4f\n" %
          (epoch, EPOCHS, running_loss, running_acc))

    # 构造临时变量
    correct, totalLoss = 0, 0
    # 关闭模型的训练状态
    net.train(False)
    # 对测试集的DataLoader进行迭代
    with torch.no_grad():
        for testImgs, labels in testDataLoader:
            testImgs = testImgs.to(device)
            labels = labels.to(device)
            outputs = net(testImgs)
            loss = lossF(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)

            # 存储测试结果
            totalLoss += loss
            correct += torch.sum(predictions == labels)

    # 计算总测试的平均准确率
    testAccuracy = correct / (BATCH_SIZE * len(testDataLoader))
    # 计算总测试的平均Loss
    testLoss = totalLoss / len(testDataLoader)
    # 将本轮结果进行可视化处理
    print("[%d/%d] Test Loss: %.4f, Test Acc: %.4f\n" %
          (epoch, EPOCHS, testLoss.item(), testAccuracy.item()))
    history['Loss'].append(running_loss)
    history['Acc'].append(running_acc)
    history['Test Accuracy'].append(testAccuracy.item())
    history['Test Loss'].append(testLoss.item())

plt.title('Loss')
plt.xlabel('epoch')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(range(EPOCHS), history['Loss'], 'red')
plt.plot(range(EPOCHS), history['Test Loss'], 'blue')
plt.legend(['Train Loss', 'Test Loss'])
plt.show()

plt.title('Accuracy')
plt.xlabel('epoch')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(range(EPOCHS), history['Acc'], 'red')
plt.plot(range(EPOCHS), history['Test Accuracy'], 'blue')
plt.legend(['Train Acc', 'Test Acc'])
plt.show()