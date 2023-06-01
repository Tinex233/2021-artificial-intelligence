import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

BATCH_SIZE = 128
LR = 0.01
EPOCH = 90
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path_train = '/root/autodl-tmp/facial-expression-recognition-main/face_images/resnet_train/emotion_images/'
path_vaild = '/root/autodl-tmp/facial-expression-recognition-main/face_images/resnet_verify/emotion_images/'

transforms_train = transforms.Compose([
    transforms.Grayscale(),#使用ImageFolder默认扩展为三通道，重新变回去就行
    transforms.RandomHorizontalFlip(),#随机翻转
    transforms.RandomRotation(degrees=15),#随机旋转
    transforms.RandomCrop((48,48),padding=None),#随机剪裁
    transforms.ColorJitter(brightness=0.5, contrast=0.5),#随机调整亮度和对比度
    transforms.ToTensor()
])
transforms_vaild = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

'''class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, 1]
        
        if self.transform:
            img = self.transform(img)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return img, label

'''
'''data_train = CustomDataset(path_train+'image_emotion.csv/', img_dir=path_train, transform=train_transforms)
data_valid = CustomDataset(path_valid+'image_emotion.csv/', img_dir=path_valid, transform=test_transforms)
'''
data_train = torchvision.datasets.ImageFolder(root=path_train,transform=transforms_train)
data_vaild = torchvision.datasets.ImageFolder(root=path_vaild,transform=transforms_vaild)

train_set = torch.utils.data.DataLoader(dataset=data_train,batch_size=BATCH_SIZE,shuffle=True)
vaild_set = torch.utils.data.DataLoader(dataset=data_vaild,batch_size=BATCH_SIZE,shuffle=False)



class ResNet(nn.Module):
    def __init__(self, *args):
        super(ResNet, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0],-1)


class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


# 残差神经网络
class Residual(nn.Module): 
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

    
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

resnet = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7 , stride=2, padding=3),
    nn.BatchNorm2d(64), 
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
resnet.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
resnet.add_module("resnet_block2", resnet_block(64, 128, 2))
resnet.add_module("resnet_block3", resnet_block(128, 256, 2))
resnet.add_module("resnet_block4", resnet_block(256, 512, 2))
resnet.add_module("global_avg_pool", GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
resnet.add_module("fc", nn.Sequential(ResNet(), nn.Linear(512, 7))) 

model = resnet
model.to(DEVICE)
optimizer = optim.SGD(model.parameters(),lr=LR,momentum=0.9)
            #optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()


train_loss = []
train_ac = []
vaild_loss = []
vaild_ac = []
y_pred = []
epoch_x = []
train_acc = []
vaild_acc = []

def train(model,device,dataset,optimizer,epoch):
    model.train()
    correct = 0
    for i,(x,y) in tqdm(enumerate(dataset)):
        x , y  = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        pred = output.max(1,keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()
        loss = criterion(output,y) 
        loss.backward()
        optimizer.step()
    
    train_acc.append(100*correct/len(data_train))
       
    train_ac.append(correct/len(data_train))   
    train_loss.append(loss.item())
    print("Epoch {} Loss {:.4f} Accuracy {}/{} ({:.0f}%)".format(epoch,loss,correct,len(data_train),100*correct/len(data_train)))

def vaild(model,device,dataset,epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i,(x,y) in tqdm(enumerate(dataset)):
            x,y = x.to(device) ,y.to(device)
            output = model(x)
            loss = criterion(output,y)
            pred = output.max(1,keepdim=True)[1]
            global  y_pred 
            y_pred += pred.view(pred.size()[0]).cpu().numpy().tolist()
            correct += pred.eq(y.view_as(pred)).sum().item()
            
    vaild_acc.append(100.*correct/len(data_vaild))
            
    vaild_ac.append(correct/len(data_vaild)) 
    vaild_loss.append(loss.item())
    print("Test Loss {:.4f} Accuracy {}/{} ({:.0f}%)".format(loss,correct,len(data_vaild),100.*correct/len(data_vaild)))


def RUN():
    for epoch in range(1,EPOCH+1):
        '''if epoch==15 :
            LR = 0.1
            optimizer=optimizer = optim.SGD(model.parameters(),lr=LR,momentum=0.9)
        if(epoch>30 and epoch%15==0):
            LR*=0.1
            optimizer=optimizer = optim.SGD(model.parameters(),lr=LR,momentum=0.9)
        '''
        
        epoch_x.append(epoch)
        
        #尝试动态学习率
        train(model,device=DEVICE,dataset=train_set,optimizer=optimizer,epoch=epoch)
        vaild(model,device=DEVICE,dataset=vaild_set,epoch=epoch)
        torch.save(model,'/root/autodl-tmp/facial-expression-recognition-main/model/model_resnet.pkl')
        
        if epoch==EPOCH:#结束生成准确率变化统计图
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy%')
            plt.plot(epoch_x,train_acc,label='Train Accuracy',color='b')
            plt.plot(epoch_x,vaild_acc,label='Test Accuracy',color='g')
            plt.plot(epoch_x,np.full(len(epoch_x), 60),color='r',linestyle='--')
            plt.legend()
            plt.savefig("/root/autodl-tmp/facial-expression-recognition-main/resnet_result.png")#,dpi=1000,bbox_inches = 'tight')
            plt.show()

if __name__ == '__main__':
    RUN()