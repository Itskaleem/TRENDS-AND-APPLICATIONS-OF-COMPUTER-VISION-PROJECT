import json
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from os.path import join, splitext
from sys import argv

def eval_svm(data_path, svm_iters=100000, svm_conf=1e-3, crossval_k=10):

    feats, labels = [], []

    with open(data_path) as f:
        annots = json.load(f)

    for annot in annots:
        
        # sometimes this happens with Wang, just skip this sample
        feat = np.asarray(annot['feat'])
        if np.any(np.isnan(feat)):
            continue
        feats.append(annot['feat'])
        labels.append(annot['label'])

    feats = np.asarray(feats, dtype=np.float32)

    # normalize features
    mean,std = np.mean(feats,axis=0), np.std(feats,axis=0)
    feats = (feats - mean) / (std + 1e-6)

    labels = np.asarray(labels, dtype=np.float32)
    # apply svm and return results
    clf = LinearSVC(max_iter=svm_iters, tol=svm_conf)
    scores = cross_val_score(clf, feats, labels, cv=crossval_k)

    return scores

if __name__ == '__main__':

    print("Evaluating on method {}".format(argv[1]))

    feat_type = argv[1]

    feat_name, ext = splitext(feat_type)
    
    assert ext == '.json', f'Expected json file, found {feat_type}'

    score = eval_svm(feat_type, crossval_k=10)
    mean_acc, std_acc = score.mean(), score.std()

    print("{:<16} : {:4.4f} {:1.4f}".format(feat_name,mean_acc,std_acc))

