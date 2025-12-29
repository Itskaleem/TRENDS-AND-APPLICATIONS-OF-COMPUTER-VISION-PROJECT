import cv2
import json
import torch
from tqdm import tqdm
from os.path import join
from os import readlink
from utils import cv_centercrop
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from sys import argv

def resnet_feats(root, type=18):

    if type == 18:    
        print("Extracting feats from pretrained ResNet18")
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    elif type == 50:
        print("Extracting feats from pretrained ResNet50")
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        raise RuntimeError(f"Model Resnet-{type} not recognized.")

    # change last layer to identity to just get the feature vector
    model.fc = torch.nn.Identity()
    model.eval()

    with open(join(root, 'annots.json')) as f:
        annots = json.load(f)

    # statistics from Imagenet
    # https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    new_annots = []
    for i in tqdm(range(len(annots))):
        path = annots[i]['path']
        x,y,w,h = annots[i]['box']

        # read and convert to greyscale
        img = cv2.imread(join(root,path))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # center crop
        img = img[int(y):int(y+h), int(x):int(x+w)]
        img = cv_centercrop(img)
        
        # convert to tensor and normalize
        img = torch.tensor(img.transpose(2,0,1),dtype=torch.float32) / 255.
        img = (img - mean.view(3,1,1)) / std.view(3,1,1)
        img = img.unsqueeze(0)

        # forward to get feature
        with torch.no_grad():
            res = model(img)

        new_annots.append({
            'id': '{:06d}'.format(annots[i]['id']),
            'feat' : res.squeeze(0).numpy().tolist(),
            'label': annots[i]['label']
        })

        break

    with open(f'resnet{type}_center.json', 'w') as f:
        json.dump(new_annots, f)


if __name__ == '__main__':

    resnet_type = argv[1]
    root = readlink('data_dlc')
    print('Computing Resnet features with center crop')
    resnet_feats(root, type=int(resnet_type))
