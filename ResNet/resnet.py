import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch.optim import lr_scheduler
# 如果网络能在GPU中训练，就使用GPU；否则使用CPU进行训练
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 这个函数包括了两个操作：将图片转换为张量，以及将图片进行归一化处理
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            #torchvision.transforms.Resize(64),
                                            torchvision.transforms.Normalize(mean=[0.5], std=[0.5])],
                                            )
path = './data/'  # 数据集下载后保存的目录

# 下载训练集和测试集
trainData = torchvision.datasets.MNIST(path, train=True, transform=transform, download=True)
testData = torchvision.datasets.MNIST(path, train=False, transform=transform)
# 设定每一个Batch的大小
BATCH_SIZE = 16
learning_rate = 0.03
# 构建数据集和测试集的DataLoader
trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE)


# 定义t网络结构
class ResidualBlock(nn.Module):
    """
    每一个ResidualBlock,需要保证输入和输出的维度不变
    所以卷积核的通道数都设置成一样
    """
    def __init__(self, channel):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

    def forward(self, x):
        """
        ResidualBlock中有跳跃连接;
        在得到第二次卷积结果时,需要加上该残差块的输入,
        再将结果进行激活,实现跳跃连接 ==> 可以避免梯度消失
        在求导时,因为有加上原始的输入x,所以梯度为: dy + 1,在1附近
        """
        y = nn.functional.relu(self.conv1(x))
        y = self.conv2(y)

        return nn.functional.relu(x + y)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.res_block_1 = ResidualBlock(16)
        self.res_block_2 = ResidualBlock(32)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = nn.functional.max_pool2d(nn.functional.relu(self.conv1(x)), 2)
        x = self.res_block_1(x)
        x = nn.functional.max_pool2d(nn.functional.relu(self.conv2(x)), 2)
        x = self.res_block_2(x)
        x = x.view(in_size, -1)
        x = self.fc1(x)
        return nn.functional.log_softmax(x, dim=1)




net = Net().to(device)

summary(net, input_size=(1, 28, 28), batch_size=-1)

lossF = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
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