from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def get_caim(sp, scheme, xi, y):
    """CAIM score for discretization
    Args:
        sp : indexes of x corresponding to each bin
        scheme : set of thresholds for the discretized bins
        xi : attribute being discretized
        y : target to be used for discretization of xi
    Returns:
        CAIM score
    """
    sp.insert(0, 0)
    sp.append(xi.shape[0])
    n = len(sp) - 1

    caim = 0
    for r in range(n):
        init = sp[r]
        fin = sp[r + 1]
        val, counts = np.unique(y[init:fin], return_counts=True)

        Mr = xi[init:fin].shape[0]
        maxr = counts.max()
        caim += (maxr / Mr) * maxr

    caim /= n
    return caim


class CAIMD(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features="auto", max_process=30, score=get_caim, **kwargs):
        """
        CAIM discretization class
        Parameters
        ----------
        categorical_features : 'auto' or 'all' or list/array of indices or list of labels
        Specify what features are treated as categorical.
        - 'auto' (default): Only those features whose number of unique values exceeds the number of classes
                            of the target variable by 2 times or more
        - array of indices: array of categorical feature indices
        - list of labels: column labels of a pandas dataframe
        Example
        ---------
        >>> from caimcaim import CAIMD
        >>> from sklearn.datasets import load_iris
        >>> iris = load_iris()
        >>> X = iris.data
        >>> y = iris.target
        >>> caim = CAIMD()
        >>> x_disc = caim.fit_transform(X, y)
        """

        self.max_process = max_process
        self.score = score

        if isinstance(categorical_features, str):
            self._features = categorical_features
            self.categorical = None
        elif (isinstance(categorical_features, list)) or (
            isinstance(categorical_features, np.ndarray)
        ):
            self._features = None
            self.categorical = categorical_features
        else:
            raise CategoricalParamException(
                "Wrong type for 'categorical_features'. Expected 'auto', an array of indicies or labels."
            )

    def parallel(self, args):
        (xj, yj, sp) = args
        scheme = self.mainscheme[:]
        scheme.append(sp)
        scheme.sort()
        c = self.score(self.index_from_scheme(scheme[1:-1], xj), scheme, xj, yj)
        return (c, scheme, sp)

    def fit(self, data, x, y):
        """
        Fit CAIM
        Parameters
        ----------
        data : dataframe containing data of x and y
        x : node to be discretized
        y : target variable
        Returns
        -------
        self
        """

        X = data[x].values.reshape(data.shape[0], -1)
        y = data[y].values.reshape(data.shape[0], -1)

        self.split_scheme = dict()
        if isinstance(X, pd.DataFrame):
            # self.indx = X.index
            # self.columns = X.columns
            if isinstance(self._features, list):
                self.categorical = [X.columns.get_loc(label) for label in self._features]
            X = X.values
            y = y.values
        if self._features == "auto":
            self.categorical = self.check_categorical(X, y)
        categorical = self.categorical
        print("Categorical", categorical)

        # multiple targets
        y_temp = np.ndarray(shape=(y.shape[0], 1))
        label_val = dict()
        num_labels = 0
        for i in range(y.shape[0]):
            if tuple(y[i]) not in label_val:
                label_val[tuple(y[i])] = num_labels
                num_labels += 1
            y_temp[i] = np.asarray(label_val[tuple(y[i])])
        y = y_temp

        min_splits = np.unique(y).shape[0]

        for j in range(X.shape[1]):
            # if j in categorical:
            #     continue
            xj = X[:, j]
            xj = xj[np.invert(np.isnan(xj))]
            new_index = xj.argsort()
            xj = xj[new_index]
            yj = y[new_index]
            allsplits = np.unique(xj)[1:-1].tolist()  # potential split points
            global_caim = -1
            self.mainscheme = [xj[0], xj[-1]]

            # label sets generated to handle multi-label
            labels = np.unique(yj)
            Y = dict()
            for x in np.unique(xj):
                Y[x] = set()
            for i in range(xj.shape[0]):
                Y[xj[i]].add(yj[i][0])
            yj = []
            for i in range(xj.shape[0]):
                temp = np.ndarray(shape=len(labels) - len(Y[xj[i]]))
                temp[:] = -1e10
                temp = np.append(temp, np.array(list(Y[xj[i]])))
                yj.append(temp)
            yj = np.asarray(yj)

            best_caim = 0
            k = 1
            while ((k <= min_splits) or (global_caim < best_caim)) and (allsplits):
                split_points = np.random.permutation(allsplits).tolist()
                best_scheme = None
                best_point = None
                best_caim = 0
                k = k + 1
                pool = Pool(min(self.max_process, len(split_points)))
                args = list(zip([xj] * len(split_points), [yj] * len(split_points), split_points))
                results = pool.map(self.parallel, args)
                pool.close()
                pool.join()
                (best_caim, best_scheme, best_point) = sorted(results, key=lambda i: i[0])[-1]
                if (k <= min_splits) or (best_caim > global_caim):
                    self.mainscheme = best_scheme
                    global_caim = best_caim
                    try:
                        allsplits.remove(best_point)
                    except ValueError:
                        raise NotEnoughPoints(
                            "The feature #"
                            + str(j)
                            + " does not have"
                            + " enough unique values for discretization!"
                            + " Add it to categorical list!"
                        )

            self.split_scheme[j] = self.mainscheme
            print("#", j, " GLOBAL CAIM ", global_caim)
        return self, global_caim

    def transform(self, X):
        """
        Discretize X using a split scheme obtained with CAIM.
        Parameters
        ----------
        X : pandas dataframe, shape [n_samples, n_features]
            Input array can contain missing values
        Returns
        -------
        X_di : sparse matrix if sparse=True else a 2-d array, dtype=int
            Transformed input.
        """
        X = X[X.columns[0]].values.reshape(X.shape[0], -1)
        if isinstance(X, pd.DataFrame):
            self.indx = X.index
            self.columns = X.columns
            X = X.values
        X_di = X.copy()
        categorical = self.categorical

        scheme = self.split_scheme
        for j in range(X.shape[1]):
            # if j in categorical:
            #     continue
            sh = scheme[j]
            sh[-1] = sh[-1] + 1
            xj = X[:, j]
            # xi = xi[np.invert(np.isnan(xi))]
            for i in range(len(sh) - 1):
                ind = np.where((xj >= sh[i]) & (xj < sh[i + 1]))[0]
                X_di[ind, j] = i
        if hasattr(self, "indx"):
            return pd.DataFrame(X_di, index=self.indx, columns=self.columns)
        return X_di

    def fit_transform(self, X, y):
        """
        Fit CAIM to X,y, then discretize X.
        Equivalent to self.fit(X).transform(X)
        """
        self.fit(X, y)
        return self.transform(X)

    def index_from_scheme(self, scheme, x_sorted):
        split_points = []
        for p in scheme:
            split_points.append(np.where(x_sorted > p)[0][0])
        return split_points

    def check_categorical(self, X, y):
        categorical = []
        ny2 = 2 * np.unique(y).shape[0]
        for j in range(X.shape[1]):
            xj = X[:, j]
            xj = xj[np.invert(np.isnan(xj))]
            if np.unique(xj).shape[0] < ny2:
                categorical.append(j)
        return categorical


class CategoricalParamException(Exception):
    # Raise if wrong type of parameter
    pass


class NotEnoughPoints(Exception):
    # Raise if a feature must be categorical, not continuous
    pass
