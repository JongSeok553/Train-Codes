import os
import tqdm
import numpy as np

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import copy
#import torchvision.datasets as datasets
import torchvision.models as models
import model as modelZoo
# 사용 가능한 torchvision 모델
# 사용 안하는거 주석 처리하고 사용하면 초기화 빠름
from torchvision.models.alexnet import alexnet
from torchvision.models.densenet import densenet121,densenet169,densenet201,densenet161
from torchvision.models.googlenet import googlenet
from torchvision.models.inception import inception_v3
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models.resnet import resnet18,resnet34,resnet50,resnet101,resnet152,resnext50_32x4d,resnext101_32x8d
from torchvision.models.shufflenetv2 import shufflenet_v2_x0_5,shufflenet_v2_x1_0,shufflenet_v2_x1_5,shufflenet_v2_x2_0
from torchvision.models.squeezenet import squeezenet1_0,squeezenet1_1
from torchvision.models.vgg import vgg11_bn,vgg13_bn,vgg16_bn,vgg19_bn
from collections import OrderedDict
from transforms import build_transforms
from folder import ImageFolder
from Config import config

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

# def train(dataset_root, pretrained_model, model_save_path, test_only=True, num_epochs=120):
def run(mode='Test'):
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path,exist_ok=True)
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mode             = config['Mode']
    start_epoch      = config['Start_Epoch']
    epochs           = config['Epochs']
    learning_rate    = config['LearningRate']
    pring_freq       = config['Print_freq']
    train_batch_size = config['Batch_size']
    eval_batch_size  = config['Batch_size']
    lr_decay_period  = config['lr_decay_period']
    input_resolution = config['Input_size']
    dataset_root     = config['DataRoot']
    transform        = config['Transform']

    pretrained_model = config['ProjectRoot'] + config['Pretrained_Model']
    using_gpus       = config['UsingGPU']
    num_workers      = config['Num_workers']

    # data loader
    traindir = os.path.join(dataset_root,'train')
    testdir = os.path.join(dataset_root,'val')
    valdir = os.path.join(dataset_root,'val')

    transform_train, transform_eval = build_transforms(height=input_resolution[0],width=input_resolution[1],transforms=transform)

    testloader = data.DataLoader(
        ImageFolder(testdir,transform=transform_eval),
        batch_size=eval_batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True
    )

    trainloader = data.DataLoader(
        ImageFolder(traindir,transform=transform_train),
        batch_size=train_batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True
    )

    valloader = data.DataLoader(
        ImageFolder(valdir,transform=transform_eval),
        batch_size=eval_batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True
    )

    # label file will created
    label_name = os.path.join(model_save_path,'label.txt')
    if os.path.isfile(label_name):
        os.remove(label_name)
    label = open(label_name,'a')
    for class_name in ImageFolder(traindir,transform=transform_train).classes:
        label.write(class_name+"\n")
    label.close()

    num_classes = len(ImageFolder(traindir,transform=transform_train).classes)

    # model create
    Net = getattr(modelZoo, "se_resnet50")
    model = Net(num_classes=num_classes)

    # DNN = resnet50
    # model = DNN(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,num_classes)
    print(model)
    print(Net)

    if torch.cuda.device_count() > 1 and mode == "Train":
        model = nn.DataParallel(model)
    model = model.to(device)

    # loss, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # load pretrained model
    if os.path.isfile(pretrained_model):
        checkpoint = torch.load(pretrained_model)
        pretrained_dict = remove_prefix(checkpoint, 'module.')
        model.load_state_dict(pretrained_dict)
        print("load pretrained model:",pretrained_model)
    else:
        if not pretrained_model == "":
            print("cannot find pretrained model:",pretrained_model)
    if mode=='Test':
        print("Test Mode")
        temp = eval(model,testloader,criterion,device,'Test',label_file_path=label_name)
        return

    # training
    total_step = len(trainloader)
    curr_lr = learning_rate
    model.train()
    best_acc = 0.0
    for epoch in range(start_epoch, epochs):
        correct = 0.0
        total = 0.0
        
        for batch_idx, (images, labels, _) in enumerate(tqdm.tqdm(trainloader, desc="Epoch [{}/{}]-train".format(epoch+1, epochs))):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()

            prediction = torch.max(outputs.data,1)[1]
            correct += prediction.eq(labels.data.view_as(prediction)).cpu().sum()
            total += labels.shape[0]

            if (batch_idx+1) % pring_freq == 0 or (batch_idx+1) == len(trainloader):
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} Acc: {:.4f}%".format(epoch+1, epochs, batch_idx+1, total_step, loss.item(), 100*correct.item()/total))
                correct = 0.0
                total = 0.0
        
        # decay learning late
        if (epoch+1) % lr_decay_period == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)

        stat = "Epoch [{}/{}], validation".format(epoch+1, epochs)

        model_backbone = config['Backbone']
        best_acc = eval(model, valloader, criterion, device, stat, label_file_path=label_name, best_acc=best_acc,
                        model_backbone=model_backbone)
        try:
            state_dict = model.module.state_dict()
        except AttributeError:
            state_dict = model.state_dict()

        # Save the model checkpoint            
        state = {
            'epoch': epoch+1,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
        }            
        save_name = os.path.join(model_save_path,model_backbone+'_'+str(epoch+1)+'.pth.tar')
        torch.save(state, save_name)

        model.train()
    
    eval(model,testloader,criterion,device,'Test',label_file_path=label_name)
    Train_History.close()

def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def eval(model,dataloader,criterion,device,stat,label_file_path, best_acc=0.0, model_backbone='se_resnet50'):
    if "Test" in stat:
        progressbar = "Test images"
        epoch = "test"
    else:
        progressbar = stat.replace(", ","-")
        epoch = "val_"+stat.split("[")[1][0]

    label_file = open(label_file_path,'r')
    class_names = label_file.readlines()
    label_file.close()

    model.eval()
    with torch.no_grad():
        test_loss = 0

        num_images_per_class = {}
        correct_per_class = {}
        # 0으로 초기화
        for i in range(len(class_names)):
            num_images_per_class[i] = 0
            correct_per_class[i] = 0

        with open(epoch+'_result.csv', 'w') as csvFile:
            contents = 'path,label,output'
            csvFile.write(contents)

            for batch_idx, (images, labels, paths) in enumerate(tqdm.tqdm(dataloader, desc=progressbar)):
                images = images.to(device)
                labels = labels.to(device)
                paths = np.asarray(paths)

                outputs = model(images)
                batch_loss = criterion(outputs, labels)
                test_loss += batch_loss.item()

                _, predicted = torch.max(outputs.data, 1)

                save = np.vstack((paths,labels.cpu().numpy()))
                save = np.vstack((save,predicted.cpu().numpy()))
                save = np.transpose(save)

                csv_rows = ["{},{},{}".format(i, j, k) for i, j, k in save]
                csv_text = "\n"+"\n".join(csv_rows)

                csvFile.write(csv_text)
                #writer.writerows(save)

                count = 0
                for label in labels:
                    label = int(label.item())
                    
                    num_images_per_class[label] += 1
                    
                    if label == predicted[count].item():
                        correct_per_class[label] += 1
                        
                    count += 1

            csvFile.close()

            correct = sum(list(correct_per_class.values()))
            total = sum(list(num_images_per_class.values()))
            stat += " Loss: {:.4f}, Acc: {:.4f}%".format(test_loss/len(dataloader), 100 * correct / total)    
            print(stat)

            epoch_acc = 100 * correct / total
            print(f'current acc {epoch_acc}, best_acc {best_acc}')
            mode = config['Mode']
            if epoch_acc > best_acc and mode !='Test':
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_model_wts)
                # save_name = os.path.join(model_save_path, model_backbone + '_' + str(epoch + 1) + '.pth.tar')
                path = os.getcwd() + "/best/" + "best_" + model_backbone + "_" + str(best_acc) + ".pth.tar"
                torch.save(model.state_dict(), path)

            for class_name in list(num_images_per_class.keys()):
                print("[{}]: {:.4f}%".format(class_names[class_name].replace("\n",""),100*correct_per_class[class_name]/num_images_per_class[class_name]))

    return best_acc
def forwardspeedTest():
    import time
    
    DNN_list = [alexnet,densenet121,densenet169,densenet201,densenet161,googlenet,mobilenet_v2,resnet18,resnet34,resnet50,resnet101,resnet152,resnext50_32x4d,resnext101_32x8d,shufflenet_v2_x0_5,shufflenet_v2_x1_0,shufflenet_v2_x1_5,shufflenet_v2_x2_0,squeezenet1_0,squeezenet1_1,vgg11_bn,vgg13_bn,vgg16_bn,vgg19_bn]

    x = torch.FloatTensor(1, 3, 224, 224).cuda()
    for DNN in DNN_list:
        model = DNN()
        model.cuda()

        T=[]
        for i in range(101):
            t1 = time.time()
            model(x)
            t2 = time.time() - t1
            if not (i==0):
                T.append(t2)

        print("{}:{:.4f}s".format(DNN.__name__,sum(T)/len(T)))

def makeONNXmodel(name,pretrained_model):
    # 사용 예시
    # pretrained_model = "model.pth.tar"
    # name = "model.onnx"
    # makeONNXmodel(name,pretrained_model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 아래 부분 원하는 모델, class 수로 수정해서 사용해야함
    num_classes = 3
    # model create
    DNN = resnet50
    model = DNN(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,num_classes)
    model = model.to(device)

    # load pretrained model
    if os.path.isfile(pretrained_model):
        checkpoint = torch.load(pretrained_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("load pretrained model:",pretrained_model)
    else:
        if not pretrained_model == "":
            print("cannot find pretrained model:",pretrained_model)

    # 입력 사이즈 변경 학습된대로 필요
    example = torch.randn(1, 3, 224, 224).to(device)

    torch.onnx.export(model, example, name, export_params=True, input_names=['images'] ,output_names=['outputs'])

def makeLibtorchmodel(name,pretrained_model):
    # 사용 예시
    # pretrained_model = "model.pth.tar"
    # name = "model.pt"
    # makeLibtorchmodel(name,pretrained_model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 아래 부분 원하는 모델, class 수로 수정해서 사용해야함
    num_classes = 3
    # model create
    DNN = resnet50
    model = DNN(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,num_classes)
    model = model.to(device)

    # load pretrained model
    if os.path.isfile(pretrained_model):
        checkpoint = torch.load(pretrained_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("load pretrained model:",pretrained_model)
    else:
        if not pretrained_model == "":
            print("cannot find pretrained model:",pretrained_model)

    # 입력 사이즈 변경 학습된대로 필요
    example = torch.randn(1, 3, 112, 112).to(device)

    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(name)

if __name__ == '__main__':
    dataset_root = '/media/ailab1/55edca0a-e68e-4999-89c4-738b020a4c10/OBR_Flame/data/'
    pretrained_model = ''
    model_save_path = 'temp'

    Train_History = open('best/histroy.txt', 'w')
    backbone = config['Backbone']
    Train_History.write(backbone + '\n')

    best_acc = 0.0
    run()
    #forwardspeedTest()
    #name = "model.onnx"
    #makeONNXmodel(name,pretrained_model)
    #name = "model.pt"
    #makeLibtorchmodel(name,pretrained_model)