import json
import cv2
import torch
import numpy as np
from os.path import join
from os import readlink
from tqdm import tqdm
from utils import cv_centercrop, cv_randomcrops
from sys import argv

def wang_features(img):
    '''
    Feature extraction from image as in Wang et al (https://www.sciencedirect.com/science/article/pii/S1742287617300439)
    '''

    k = np.asarray([
        [0., 0.5, 0.],
        [0., 0. , 0.],
        [0., 0.5, 0.]
    ])

    # filter, compute residual and crop first and last rows and columns
    filt_y = cv2.filter2D(img, ddepth=-1, kernel=k)
    filt_x = cv2.filter2D(img, ddepth=-1, kernel=k.T)
    res_x = (img-filt_x)[1:-1,1:-1]
    res_y = (img-filt_y)[1:-1,1:-1]

    feats = []
    # for both vertical and horizontal residual
    for res in [res_x,res_y]:

        # compute patches
        unfold = torch.nn.Unfold(kernel_size=5, dilation=1, padding=0, stride=1)
        res = torch.tensor(res,dtype=torch.float)
        res = res.unsqueeze(0).unsqueeze(1)
        patches = unfold(res)[0]

        # reshape and extract centers
        patches = patches.reshape((5,5,-1))
        corr_coff = torch.zeros((5,5))
        p_center = patches[2,2,:]

        # copmute correlation coefficient with all other vectors
        for i in range(5):
            for j in range(5):
                p_ij = patches[i,j,:]
                corr_coff[i,j] = torch.corrcoef(torch.stack((p_center, p_ij),dim=0))[0,1]
        
        # get upper triangular matrix (specific for 5x5)
        feat = torch.cat((corr_coff[0,:],corr_coff[1,1:],corr_coff[2,3:], corr_coff[3,3:], corr_coff[4,4].unsqueeze(0)))
        feats.append(feat)

    feats = torch.cat(feats)

    return feats

def wang(data_path, crop_center=False):

    with open(join(data_path,'annots.json')) as f:
        annots = json.load(f)

    new_annots = []
    for i in tqdm(range(len(annots))):
        path = annots[i]['path']
        x,y,w,h = annots[i]['box']
        
        # read and convert to greyscale
        img = cv2.imread(join(data_path,path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # crop to document bbox
        img = img[int(y):int(y+h), int(x):int(x+w)]

        # optionally crop center
        if crop_center:
            img = cv_centercrop(img)

        # get and save features
        feat = wang_features(img)
        new_annots.append({
            'id': '{:06d}'.format(annots[i]['id']),
            'feat' : feat.numpy().tolist(),
            'label': annots[i]['label']
        })

        break
    
    if crop_center:
        filename = 'wang_center.json'
    else:
        filename = 'wang.json'

    with open(filename, 'w') as f:
        json.dump(new_annots, f)

def wang_multiple(data_path):

    with open(join(data_path, 'annots.json')) as f:
        annots = json.load(f)

    new_annots = []
    for i in tqdm(range(len(annots))):
        path = annots[i]['path']
        x,y,w,h = annots[i]['box']
        
        # read and convert to greyscale        
        img = cv2.imread(join(data_path,path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # crop document, generate 5 random crops
        img = img[int(y):int(y+h), int(x):int(x+w)]
        imgs = cv_randomcrops(img, size=(224,224),num=5)

        # compute feat and save for each
        for j,img in enumerate(imgs):
            feat = wang_features(img)
            
            new_annots.append({
                'id': '{:06d}_{}'.format(annots[i]['id'],j),
                'feat' : feat.numpy().tolist(),
                'label': annots[i]['label']
            })

        break

    with open('wang_multiple.json', 'w') as f:
        json.dump(new_annots, f)

if __name__ == '__main__':

    wang_type = argv[1]
    root = readlink('data_dlc')

    if wang_type == 'center':
        print('Computing Wang et al features with center crop')
        wang(root, crop_center=True)
    
    elif wang_type == 'standard':
        print('Computing Wang et al features with full document')
        wang(root, crop_center=False)    
    
    elif wang_type == 'multiple':
        print('Computing Wang et al features with 5 crops per image')
        wang_multiple(root)
    
    else:
        raise RuntimeError(f"Wang type {wang_type} not supported.")

