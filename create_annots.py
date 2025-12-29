import json
import subprocess
from os import readlink, listdir
from os.path import join


def make_annots(root):

    id = 0
    annots = []
    for type in ['or','re']:

        for doc_type in listdir(join(root,type,'images')):

            for split in listdir(join(root,type,'images',doc_type)):

                img_path = join(root,type,'images',doc_type,split)
                annot_path = join(root,type,'annotations',doc_type,split + '.json')

                with open(annot_path) as f:
                    split_annot = json.load(f)

                for img in listdir(img_path):
                    
                    if '(' in img:
                        subprocess.call('rm "{}"'.format(join(img_path,img)),shell=True)
                        continue

                    found=False
                    for metadata in split_annot['_via_img_metadata'].values():
                        if metadata['filename'] == img:
                            # found the crop finally
                            xs = metadata['regions'][0]['shape_attributes']['all_points_x']
                            ys = metadata['regions'][0]['shape_attributes']['all_points_y']
                            found=True
                            break
                    
                    if not found:
                        raise RuntimeError("image {} not found in annot file {}".format(img,annot_path))
                    
                    save_path = join(type,'images',doc_type,split,img)
                    min_w,min_h = max(0, min(xs)), max(0,min(ys))
                    w,h = max(xs)-min_w, max(ys)-min_h
                    annots.append({
                        'id'  : id,
                        'path': save_path,
                        'box' : [min_w,min_h,w,h],
                        'label' : 1 if type == 're' else 0
                    })
                    id += 1

    with open(join(root, 'annots.json'),'w') as f:
        json.dump(annots,f)

if __name__ == '__main__':

    root = readlink('data_dlc')
    make_annots(readlink('data_dlc'))
