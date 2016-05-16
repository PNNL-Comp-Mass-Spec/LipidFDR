# python svm_training.py -i LiquidTrainingData/Negative/\[GP0401\].txt -o training_neg/GP0401
import sys
import os
import getopt
import numpy as np
from sklearn import svm
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import pandas as pd


# Utility function to move the midpoint of a colormap to be around
# the values of interest.
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def main(argv):
    inputfile = ''
    outputfile = ''
    skip = False
    try:
        opts, args = getopt.getopt(argv, "hi:o:s:", ["ifile=", "ofile=", "skip="])
    except getopt.GetoptError:
        print ('test.py -i <inputfile> -o <outputfile-prefix> -s <skip-visualization>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('test.py -i <inputfile> -o <outputfile-prefix> -s <skip-visualization>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-s", "--skip"):
            skip = (arg in ("t", "T", "True", "true"))
    print ('Input file is "' + inputfile)
    print ('Output file is "' + outputfile)
    print ('Skip visualization? %s' % skip)
    training(inputfile, outputfile, skip)


def training(inputfile, outputfile, skip_visualization=False):
    try:
        os.mkdir(outputfile)
    except OSError:
        print(outputfile + ': this folder already exists')
    # load training data
    data = pd.read_table(inputfile)
    data = np.array([data['RT'],
                     data['Score'],
                     data['Cosine Score'],
                     data['Cosine M-1 Score'],
                     data['Class']])

    num_features = 4
    cols = ['RT', 'Score', 'Cosine Score', 'Cosine M-1 Score']
    training_data = np.transpose(data)

    # delete duplicates
    size = training_data.shape[0]
    training_data = np.unique(np.ascontiguousarray(training_data).view(np.dtype((np.void, training_data.dtype.itemsize*training_data.shape[1])))).view(training_data.dtype).reshape(-1, training_data.shape[1])
    print("%d duplicates are deleted" % (size - training_data.shape[0]))

    # data normalization
    norm_avg = np.mean(training_data[:, :num_features], axis=0)
    print(norm_avg)
    norm_st = np.std(training_data[:, :num_features], axis=0)
    print(norm_st)
    training_data[:, :num_features] = (training_data[:, :num_features] - norm_avg) / norm_st

    np.savetxt(outputfile + '/_norm.txt', np.array([norm_avg, norm_st]), delimiter='\t')

    # SVM - rbf kernel : Training
    X = training_data[:, 0:num_features]
    y = training_data[:, num_features]

    # Train classifiers
    C_range = np.logspace(-2, 2, 5)
    gamma_range = np.logspace(-2, 2, 5)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(y, n_iter=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(svm.SVC(probability=True), param_grid=param_grid, cv=cv)
    grid.fit(X, y)

    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))
    f = open(outputfile + '/_param.txt', 'w+')
    f.write("The best parameters are %s with a score of %0.2f"
            % (grid.best_params_, grid.best_score_))
    f.close()

    # Model persistence
    from sklearn.externals import joblib
    joblib.dump(grid.best_estimator_, outputfile + '/_svmmodel.pkl')

    if not skip_visualization:
        # plot the scores of the grid
        # grid_scores_ contains parameter settings and scores
        # We extract just the scores
        scores = [x[1] for x in grid.grid_scores_]
        scores = np.array(scores).reshape(len(C_range), len(gamma_range))

        # Draw heatmap of the validation accuracy as a function of gamma and C
        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
                   norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
        plt.xlabel('gamma')
        plt.ylabel('C')
        plt.colorbar()
        plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
        plt.yticks(np.arange(len(C_range)), C_range)
        plt.title('Validation accuracy')
        plt.savefig(outputfile + '/_param.png')

    dist = grid.best_estimator_.decision_function(X)
    prob = grid.best_estimator_.predict_log_proba(X)
    pre = grid.best_estimator_.predict(X)
    print(dist)
    print(prob)
    print(pre)

    # View scatter plot for all dimensions
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(num_features - 1, num_features - 1)
    for i in range(0, num_features - 1):
        for j in range(i, num_features - 1):
            ax = fig.add_subplot(gs[i * (num_features - 1) + j])
            ax.scatter(X[y > 0, j + 1], X[y > 0, i], color='r', label='TrainVerified', alpha=0.5)
            ax.scatter(X[y == 0, j + 1], X[y == 0, i], color='b', label='TrainDecoy', alpha=0.5)
            ax.set_xlabel(cols[j + 1], fontsize=20)
            ax.set_ylabel(cols[i], fontsize=20)
    plt.tight_layout()
    plt.savefig(outputfile + '/_scatter.png')

    # View histogram of permutation scores
    plt.figure(figsize=(8, 6))

    plt.subplot(2, 1, 1)
    plt.hold(True)
    plt.scatter(prob[pre > 0, 0], prob[pre > 0, 1], color='r', label='Verified', alpha=0.5)
    plt.scatter(prob[pre == 0, 0], prob[pre == 0, 1], color='b', label='Decoy', alpha=0.5)
    plt.title('Probability')
    plt.ylabel('Probability of Verified')
    plt.xlabel('Probability of Decoy')
    plt.hold(False)

    plt.subplot(2, 1, 2)
    plt.hold(True)
    plt.hist(dist[pre > 0], 50, color='r', alpha=0.5)
    plt.hist(dist[pre == 0], 50, color='b', alpha=0.5)
    plt.xlabel('SVM Score')
    plt.xlabel('Frequency')
    plt.hold(False)
    plt.savefig(outputfile + '/_histo.png')

if __name__ == "__main__":
    main(sys.argv[1:])
