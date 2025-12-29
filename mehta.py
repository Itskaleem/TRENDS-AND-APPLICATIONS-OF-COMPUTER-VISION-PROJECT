import json
import cv2
import math
import torch
import numpy as np
from os.path import join
from os import readlink
from tqdm import tqdm
from torch.nn.functional import conv2d
from utils import cv_centercrop, crop_divisible
from sys import argv


def all_near(root, aliasing_filter=False, threshold=1., crop_center=False):

    with open(join(root, 'annots.json')) as f:
        annots = json.load(f)

    root_2 = math.sqrt(2.)

    weights = torch.stack([
        torch.tensor([
            [1.,1.],
            [-1.,-1.]
        ]),
        torch.tensor([
            [1.,-1.],
            [1.,-1.]
        ]),
        torch.tensor([
            [0.,root_2],
            [-root_2,0.]
        ]),
        torch.tensor([
            [root_2,0.],
            [0.,-root_2]
        ]),
        torch.tensor([
            [2.,-2.],
            [-2.,2.]
        ]),
    ],dim=0)
    weights = weights.view(5, 1, 2, 2).to(torch.float)

    new_annots = []

    for i in tqdm(range(len(annots))):
        path = annots[i]['path']
        x,y,w,h = annots[i]['box']
        
        # read, crop by box and convert to greyscale
        img = cv2.imread(join(root,path))
        img = img[int(y):int(y+h), int(x):int(x+w)]
        

        if crop_center:
            # crop only center (already divisible by 8)
            img = cv_centercrop(img)
            size_h,size_w = 224 // 4, 224 // 4
        else:
            # crop all image
            img = crop_divisible(img, divisor=8)
            size_h, size_w = img.shape[0] // 4, img.shape[1] // 4

        # apply aliasing filter
        if aliasing_filter:
            img = cv2.medianBlur(img,ksize=3)

        # apply canny detector
        img = cv2.Canny(img,100,200)

        # make 4x4 patched
        unfold = torch.nn.Unfold(kernel_size=(size_h,size_w), dilation=1, padding=0, stride=(size_h,size_w))
        img = torch.tensor(np.asarray(img,dtype=float) /255.,dtype=torch.float).unsqueeze(0).unsqueeze(1)
        patches = unfold(img)[0]
        patches = patches.reshape((-1, 1, size_h, size_w))

        # convolve patches with 2x2 filter defined above
        output = conv2d(patches, weights, bias=None, stride=2)
        # define bins and descriptor
        bins = output.reshape(16,5,-1)
        desc = torch.zeros(16,5)
        
        # iterate over each of the 4x4 image block
        for j in range(16):
            # get max value and most present edge
            idxs = torch.zeros(5).to(torch.long)
            maxs, argmaxs = torch.max(bins[j],dim=0)
            # filter by threshold
            idx, max_b = torch.unique(argmaxs[maxs>threshold],return_counts=True)
            # update bins
            idxs[idx] = max_b
            desc[j] = idxs

        # compute globals and concat all
        desc_loc = torch.mean(desc,dim=0)
        desc_glob = torch.sum(desc,dim=0)
        desc_all = torch.sum(desc)
        final_desc = torch.cat((desc.flatten(), desc_loc, desc_glob, desc_all.unsqueeze(0)),dim=0)

        new_annots.append({
            'id': '{:06d}'.format(annots[i]['id']),
            'feat' : final_desc.numpy().tolist(),
            'label': annots[i]['label']
        })

        break

    filename = 'mehta'
    if aliasing_filter:
        filename = filename + '_filter'
    filename = filename + '.json'

    with open(filename, 'w') as f:
        json.dump(new_annots, f)


if __name__ == '__main__':

    if len(argv) > 1 and argv[1] == 'filter':
        print("Computing Mehta et al features with anti-aliasing filter")
        all_near(readlink('data_dlc'), aliasing_filter=True, threshold=1., crop_center=False)
    else:
        print("Computing Mehta et al features without anti-aliasing filter")
        all_near(readlink('data_dlc'), aliasing_filter=False, threshold=1., crop_center=False)

