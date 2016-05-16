# python svm_classify.py -d 2015_Nov_raw/NEG_decoy/Diacylglycerophosphoinositols\ \[GP0601\].txt -t 2015_Nov_raw/NEG_target/Diacylglycerophosphoinositols\ \[GP0601\].txt -m training_neg/GP0601
import sys
import os
import getopt
import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import pandas as pd


def main(argv):
    decoy_file = ''
    target_file = ''
    model = ''
    output = ''
    try:
        opts, args = getopt.getopt(argv, "hd:t:m:o:", ["decoy=", "target=", "model=", "output="])
    except getopt.GetoptError:
        print ('test.py -d <decoy-file> -t <target-file> -m <model-prefix> -o <output dir>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('test.py -d <decoy-file> -t <target-file> -m <model-prefix> -o <output dir>')
            sys.exit()
        elif opt in ("-d", "--decoy"):
            decoy_file = arg
        elif opt in ("-t", "--target"):
            target_file = arg
        elif opt in ("-m", "--model"):
            model = arg
        elif opt in ("-o", "--output"):
            output = arg
    print ('decoy file: "' + decoy_file)
    print ('target file: "' + target_file)
    print ('model: "' + model)
    print ('output: "' + output)
    classify(decoy_file, target_file, model, output)


def classify(decoy_file, target_file, model, output):
    try:
        # load a SVM model
        rbf_svc = joblib.load(model + '/_svmmodel.pkl')
    except Exception:
        print('empty model')
        return
    # load parameters for normalization
    with open(model + '/_norm.txt', 'r') as f:
        params = np.array([np.array(row.split('\t')) for row in f.readlines()], dtype='float')
    f.close()

    num_features = 4
    # load new decoy data
    decoy_raw = pd.read_table(decoy_file)
    decoy = np.array([decoy_raw['RT'],
                      decoy_raw['Score'],
                      decoy_raw['Cosine Score'],
                      decoy_raw['Cosine M-1 Score'],
                      decoy_raw.index])
    decoy = np.transpose(decoy)
    # load new target data
    target_raw = pd.read_table(target_file)
    target = np.array([target_raw['RT'],
                       target_raw['Score'],
                       target_raw['Cosine Score'],
                       target_raw['Cosine M-1 Score'],
                       target_raw.index])
    target = np.transpose(target)

    # data standardizing
    decoy[:, :num_features] = (decoy[:, :num_features] - params[0, :]) / params[1, :]
    target[:, :num_features] = (target[:, :num_features] - params[0, :]) / params[1, :]

    ##############################################################################
    # View histogram of permutation scores
    decoy_dist = rbf_svc.decision_function(decoy[:, :num_features])
    b = np.zeros((decoy.shape[0], decoy.shape[1] + 1))
    b[:, :-1] = decoy
    b[:, -1] = decoy_dist
    decoy = b
    decoy_raw['SVM Score'] = decoy_dist
    decoy_raw.to_csv(output + '/_decoy_output.txt', sep="\t")

    target_dist = rbf_svc.decision_function(target[:, :num_features])
    b = np.zeros((target.shape[0], target.shape[1] + 2))
    b[:, :-2] = target
    b[:, -2] = target_dist
    target = b
    fdr = calFDR(target[:, -2], decoy[:, -1])
    np.savetxt(output + '/fdr.txt', fdr, delimiter='\t')
    for i in range(0, target.shape[0]):
        target[i, -1] = fdr['fdr'][fdr['dist'] == target[i, -2]][0]
    target_raw['SVM Score'] = target[:, -2]
    target_raw['FDR'] = target[:, -1]
    target_raw.to_csv(output + '/_target_output.txt', sep="\t")

    plt.figure(figsize=(12, 12))

    plt.subplot(2, 1, 1)
    plt.hold(True)
    binNum = 100.0
    dist = np.unique(np.concatenate((target_dist, decoy_dist)))
    binwidth = (max(dist) - min(dist)) / binNum

    plt.hist(target_dist, bins=np.arange(min(dist), max(dist) + binwidth, binwidth),
             color='r', edgecolor='r', alpha=0.3, label='Target')
    plt.hist(decoy_dist, bins=np.arange(min(dist), max(dist) + binwidth, binwidth),
             color='b', edgecolor='b', alpha=0.3, label='Decoy')
    plt.xlabel('SVM Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.hold(False)

    # FDR
    plt.subplot(2, 1, 2)
    plt.hold(True)
    plt.plot(fdr['dist'], fdr['target'], 'r', label='P[Target>x]')
    plt.plot(fdr['dist'], fdr['decoy'], 'b--', label='P[Decoy>x]')
    plt.plot(fdr['dist'], fdr['fdr'], 'k', label='FDR')
    plt.xlabel('SVM Score')
    plt.legend()
    plt.hold(False)

    # plt.subplot(3, 1, 3)
    # plt.hold(True)
    # plt.plot(fdr['decoy'], fdr['target'], 'k')
    # plt.ylabel('True positive rate')
    # plt.xlabel('False positive rate')
    # plt.hold(False)

    plt.tight_layout()
    plt.savefig(output + '/_model_use.png')


def calFDR(target, decoy):
    scores = np.unique(np.concatenate((target, decoy)))
    scores = scores[::-1]

    fdr = np.zeros((scores.shape[0],),
                   dtype=[('target', 'f4'), ('decoy', 'f4'), ('fdr', 'f4'), ('dist', 'f4')])
    for i in range(0, scores.shape[0]):
        s = scores[i]
        nt = np.where(target >= s)[0].shape[0] / target.shape[0]
        nd = np.where(decoy >= s)[0].shape[0] / decoy.shape[0]
        if nt == 0 or (nd / nt) > 1.0:
            fdr[i, ] = (nt, nd, 1.0, s)
        else:
            fdr[i, ] = (nt, nd, nd / nt, s)
    return fdr

if __name__ == "__main__":
    main(sys.argv[1:])
