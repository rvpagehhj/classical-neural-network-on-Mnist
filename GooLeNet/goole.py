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
class InceptionModule(torch.nn.Module):
    def __init__(self,channels_in):
        super().__init__()
        self.layer1_kenel1_avg=torch.nn.AvgPool2d(3,stride=1,padding=1)
        self.layer1_kenel2_1by1=torch.nn.Conv2d(channels_in,24,1)
        self.layer2_kenel1_1by1=torch.nn.Conv2d(channels_in,16,1)
        self.layer3_kenel1_1by1=torch.nn.Conv2d(channels_in,16,1)
        self.layer3_kenel2_5by5=torch.nn.Conv2d(16,24,5,padding=2)
        self.layer4_kenel1_1by1=torch.nn.Conv2d(channels_in,16,1)
        self.layer4_kenel2_3by3=torch.nn.Conv2d(16,23,3,padding=1)
        self.layer4_kenrl3_3by3=torch.nn.Conv2d(23,24,3,padding=1)

    def forward(self,x):
        x_layer1=self.layer1_kenel2_1by1(self.layer1_kenel1_avg(x))
        x_layer2=self.layer2_kenel1_1by1(x)
        x_layer3=self.layer3_kenel2_5by5(self.layer3_kenel1_1by1(x))
        x_layer4=self.layer4_kenrl3_3by3(self.layer4_kenel2_3by3(self.layer4_kenel1_1by1(x)))
        output=torch.cat([x_layer1,x_layer2,x_layer3,x_layer4],dim=1)
        return output

#GoogleNet搭建
class GoogleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=torch.nn.Conv2d(1,10,5)
        self.conv2=torch.nn.Conv2d(88,20,5)
        self.incep1=InceptionModule(channels_in=10)
        self.incep2=InceptionModule(channels_in=20)
        self.maxpool=torch.nn.MaxPool2d(2)
        self.fully_connection=torch.nn.Linear(1408,10)

    def forward(self,x):
        size_fc=x.shape[0]
        x=nn.functional.relu(self.maxpool(self.conv1(x)))
        x=self.incep1(x)
        x=nn.functional.relu(self.maxpool(self.conv2(x)))
        x=self.incep2(x)
        #print(x.shape,size_fc)
        x=x.view(size_fc,-1)
        #print(x.shape)
        x=self.fully_connection(x)
        return x



net = GoogleNet().to(device)

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