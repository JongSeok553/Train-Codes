from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import torch.onnx

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def makeONNXmodel(name, pretrained_model):
    # 사용 예시
    # pretrained_model = "model.pth.tar"
    # name = "model.onnx"
    # makeONNXmodel(name,pretrained_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 아래 부분 원하는 모델, class 수로 수정해서 사용해야함
    num_classes = 3
    # model create
    DNN = models.resnet50(pretrained=False)
    model = DNN

    # model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,num_classes)
    model = model.to(device)

    # load pretrained model
    if os.path.isfile(pretrained_model):
        checkpoint = torch.load(pretrained_model)
        # model.load_state_dict(checkpoint)
        pretrained_dict = remove_prefix(checkpoint, 'module.')
        model.load_state_dict(pretrained_dict)
        print("load pretrained model:",pretrained_model)
    else:
        if not pretrained_model == "":
            print("cannot find pretrained model:",pretrained_model)

    # 입력 사이즈 변경 학습된대로 필요
    dummy_input = torch.randn(1, 3, 112, 112).to(device)

    # Single Batch
    # torch.onnx.export(model, dummy_input, name, export_params=True, input_names=['images'] ,output_names=['outputs'])
    # Dynamic Batch
    torch.onnx.export(model, dummy_input, name, export_params=True, input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

if __name__ == "__main__":
    model_path = "/media/devbox/3e9fb723-6ca4-4420-b61a-0467facd974a/dataset/Safety_Helmet/bestmodel/"
    model_name = "best_40.pth"
    pth_model  = model_path + model_name

    onnx_dir   = "/media/devbox/3e9fb723-6ca4-4420-b61a-0467facd974a/dataset/Safety_Helmet/ModelBackup/"
    onnx_name  = "obr_safety_helmet_c3_resnet50.112x112_v1.5.0.onnx"
    onnx       = onnx_dir + onnx_name

    makeONNXmodel(onnx, pth_model)