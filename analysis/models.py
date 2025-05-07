from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np


SCORES = [
    "is_reconstructed",
    "geom_artefact",
    "recon_artefact",
    "noise",
    "intensity_gm",
    "intensity_dgm",
]


class IsRecoThenClassify(ClassifierMixin, BaseEstimator):

    def __init__(
        self, target="qcglobal", is_reco="is_reconstructed", threshold=1.0
    ):
        self.is_reco = is_reco
        self.target = target
        self.threshold = threshold
        self.is_reco_clf = RandomForestClassifier()
        self.target_clf = RandomForestClassifier()

    def fit(self, X, train_y):
        tr_y_is_reco = train_y[self.is_reco].astype(int)
        tr_y_target = train_y[self.target] > self.threshold
        self.is_reco_clf.fit(X, tr_y_is_reco)
        idx_in = tr_y_is_reco > 0
        self.X_ = X.loc[idx_in]
        self.y_ = tr_y_target.loc[idx_in]
        self.target_clf.fit(self.X_, self.y_)
        return self

    def predict(self, X):
        check_is_fitted(self)
        is_reco = self.is_reco_clf.predict(X)
        pred = np.zeros(X.shape[0])
        idx_in = is_reco > 0
        pred[idx_in] = self.target_clf.predict(X.loc[idx_in])
        return pred


class IsRecoThenClassifyV2(ClassifierMixin, BaseEstimator):

    def __init__(
        self, target="qcglobal", scores=SCORES, is_reco="is_reconstructed", threshold=1.0, iqms=None,
    ):
        self.is_reco = is_reco
        self.scores = scores
        if is_reco in self.scores:
            self.scores.remove(is_reco)
        self.target = target
        self.threshold = threshold
        self.is_reco_clf = RandomForestClassifier()
        self.scores_clf = [RandomForestRegressor() for _ in range(len(scores))]
        self.target_clf = RandomForestClassifier()

    def fit(self, X, train_y, weight=None):
        tr_y_is_reco = train_y[self.is_reco].astype(int)

        tr_y_target = train_y[self.target] > self.threshold
        self.is_reco_clf.fit(X, tr_y_is_reco, weight)
        # Train the rest on only samples with is_reconstructed > 0
        idx_in = tr_y_is_reco > 0
        X_ = X.loc[idx_in]
        weight_ = weight[idx_in] if weight is not None else None
        y_ = tr_y_target.loc[idx_in]
        train_y_ = train_y.loc[idx_in]
        for i, score in enumerate(self.scores):
            tr_y_score = train_y[score].astype(float)
            self.scores_clf[i].fit(X_, y_, weight_)
        yscores = np.array(train_y_[self.scores])
        X_ = np.concatenate((X_, yscores), axis=1)
        self.target_clf.fit(X_, y_, weight_)
        return self

    def predict(self, X):
        #check_is_fitted(self)
        is_reco = self.is_reco_clf.predict(X)

        pred = np.zeros(X.shape[0])
        idx_in = is_reco > 0
        X_ = X.loc[idx_in]
        X_scores = np.zeros((X_.shape[0], len(self.scores)))
        for i, score in enumerate(self.scores):
            X_scores[:, i] = self.scores_clf[i].predict(X_)
        
        X_ = np.concatenate((X_, X_scores), axis=1)
        pred[idx_in] = self.target_clf.predict(X_)
        return pred

    def predict_proba(self, X):
        #check_is_fitted(self)
        is_reco = self.is_reco_clf.predict_proba(X)

        
        if is_reco.shape[1] == 2:
            pred_proba = is_reco
            idx_in = is_reco[:,1] > 0.5
        else:
            pred_proba = np.zeros((X.shape[0], 2))
            pred =  self.is_reco_clf.predict(X) > 0
            if pred.sum() == 0:
                pred_proba[:,0] = 1
            else:
                pred_proba[:,1] = 1
            idx_in = pred > 0
        X_ = X.loc[idx_in]
        X_scores = np.zeros((X_.shape[0], len(self.scores)))
        for i, score in enumerate(self.scores):
            X_scores[:, i] = self.scores_clf[i].predict(X_)
        
        X_ = np.concatenate((X_, X_scores), axis=1)
        pred_proba[idx_in] = self.target_clf.predict_proba(X_)
        return pred_proba


class PredScoresThenClassify(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        target="qcglobal",
        scores=SCORES,
        threshold=1.0,
        iqms=None,
        fit_on_train=True,
    ):
        if target in scores:
            scores.remove(target)
        self.scores = scores
        self.target = target
        self.threshold = threshold
        self.scores_clf = [RandomForestRegressor() for _ in range(len(scores))]
        self.target_clf = RandomForestClassifier()
        self.iqms = iqms

    def fit(self, X, train_y):
        if self.iqms is not None:
            X = X[self.iqms]
        tr_y_target = train_y[self.target] > self.threshold
        for i, score in enumerate(self.scores):
            tr_y_score = train_y[score].astype(float)
            self.scores_clf[i].fit(X, tr_y_score)

        self.X_ = np.array(train_y[self.scores])
        
        self.y_ = tr_y_target
        self.target_clf.fit(self.X_, self.y_)
        return self

    def predict(self, X):
        check_is_fitted(self)
        if self.iqms is not None:
            X = X[self.iqms]
        pred = np.zeros(X.shape[0])
        X_scores = np.zeros((X.shape[0], len(self.scores)))
        for i, score in enumerate(self.scores):
            X_scores[:, i] = self.scores_clf[i].predict(X)
        pred = self.target_clf.predict(X_scores)
        return pred


class PredScoresThenClassifyV2(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        target="qcglobal",
        scores=SCORES,
        threshold=1.0,
        iqms=None,
        val_size=0.5,
        random_state=42,
    ):

        if target in scores:
            scores.remove(target)
        self.scores = scores
        self.target = target
        self.threshold = threshold
        self.scores_clf = [RandomForestRegressor() for _ in range(len(scores))]
        self.target_clf = RandomForestClassifier()
        self.iqms = iqms
        self.val_size = val_size
        self.random_state = random_state

    def fit(self, X, train_y, weight=None):
        from sklearn.model_selection import train_test_split

        if self.iqms is not None:
            X = X[self.iqms]
        # Split X and train_y in two groups as 80 % for training and 20 % for testing
        idx_tr, idx_te = train_test_split(np.arange(X.shape[0]),
            test_size=self.val_size, random_state=self.random_state
        )

        X_tr = X.iloc[idx_tr]
        X_te = X.iloc[idx_te]
        y_tr = train_y.iloc[idx_tr]
        y_te = train_y.iloc[idx_te]
        weight_tr = weight[idx_tr] if weight is not None else None
        weight_te = weight[idx_te] if weight is not None else None
        for i, score in enumerate(self.scores):
            tr_y_score = y_tr[score].astype(float)
            self.scores_clf[i].fit(X_tr, tr_y_score, weight_tr)

        pred_scores = np.zeros((X_te.shape[0], len(self.scores)))
        for i, score in enumerate(self.scores):
            pred_scores[:, i] = self.scores_clf[i].predict(X_te)
        self.target_clf.fit(pred_scores, y_te[self.target] > self.threshold, weight_te)
        return self

    def predict(self, X):
        if self.iqms is not None:
            X = X[self.iqms]
        # check_is_fitted(self)
        pred = np.zeros(X.shape[0])
        X_scores = np.zeros((X.shape[0], len(self.scores)))
        for i, score in enumerate(self.scores):
            X_scores[:, i] = self.scores_clf[i].predict(X)
        pred = self.target_clf.predict(X_scores)
        return pred

    def predict_proba(self, X):
        if self.iqms is not None:
            X = X[self.iqms]
        # check_is_fitted(self)
        pred = np.zeros(X.shape[0])
        X_scores = np.zeros((X.shape[0], len(self.scores)))
        for i, score in enumerate(self.scores):
            X_scores[:, i] = self.scores_clf[i].predict(X)
        return self.target_clf.predict_proba(X_scores)
    
class Interp2models(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        model1,
        model2,
        weight,
        target="qcglobal",
        scores=SCORES,
        threshold=1.0,
        iqms=None,
        
    ):

        if target in scores:
            scores.remove(target)
        self.scores = scores
        self.target = target
        self.threshold = threshold
        self.model1 = model1
        self.model2 = model2
        self.weight = weight
        self.iqms = iqms

    def fit(self, X, train_y):
        if self.iqms is not None:
            X = X[self.iqms]
        self.model1.fit(X, train_y)
        self.model2.fit(X, train_y)
        return self

    def predict(self, X):
        if self.iqms is not None:
            X = X[self.iqms]
        pred1 = self.model1.predict_proba(X)
        pred2 = self.model2.predict_proba(X)
        pred = self.weight * pred1 + (1 - self.weight) * pred2
            
        return pred[:,1] > 0.5

    def predict_proba(self, X):
        if self.iqms is not None:
            X = X[self.iqms]
        pred1 = self.model1.predict_proba(X)
        pred2 = self.model2.predict_proba(X)
        pred = self.weight * pred1 + (1 - self.weight) * pred2
            
        return pred 

class PredScoresThenClassifyV3(ClassifierMixin, BaseEstimator):
    def __init__(self, target="qcglobal", scores=SCORES, threshold=1.0, iqms=None):
        if target in scores:
            scores.remove(target)
        self.scores = scores
        self.target = target
        self.threshold = threshold
        self.scores_clf = [RandomForestRegressor() for _ in range(len(scores))]
        self.target_clf = RandomForestClassifier()
        self.iqms = iqms

    def fit(self, X, train_y):
        if self.iqms is not None:
            X = X[self.iqms]
        tr_y_target = train_y[self.target] >= self.threshold
        for i, score in enumerate(self.scores):
            tr_y_score = train_y[score].astype(float)
            self.scores_clf[i].fit(X, tr_y_score)

        yscores = np.array(train_y[self.scores])
        self.X_ = np.concatenate((X, yscores), axis=1)
        self.y_ = tr_y_target
        self.target_clf.fit(self.X_, self.y_)
        return self

    def predict(self, X):

        check_is_fitted(self)
        if self.iqms is not None:
            X = X[self.iqms]
        pred = np.zeros(X.shape[0])
        X_scores = np.zeros((X.shape[0], len(self.scores)))
        for i, score in enumerate(self.scores):
            X_scores[:, i] = self.scores_clf[i].predict(X)
        X = np.concatenate((X, X_scores), axis=1)
        pred = self.target_clf.predict(X)
        return pred


class PredScoresThenRegress(ClassifierMixin, BaseEstimator):
    def __init__(self, target="qcglobal", scores=SCORES):
        if target in scores:
            scores.remove(target)
        self.scores = scores
        self.target = target
        self.scores_clf = [RandomForestRegressor() for _ in range(len(scores))]
        self.target_clf = RandomForestRegressor()

    def fit(self, X, train_y):
        tr_y_target = train_y[self.target]
        for i, score in enumerate(self.scores):
            tr_y_score = train_y[score].astype(float)
            self.scores_clf[i].fit(X, tr_y_score)

        yscores = np.array(train_y[self.scores])
        self.X_ = np.concatenate((X, yscores), axis=1)
        self.y_ = tr_y_target
        self.target_clf.fit(self.X_, self.y_)
        return self

    def predict(self, X):

        check_is_fitted(self)
        pred = np.zeros(X.shape[0])
        X_scores = np.zeros((X.shape[0], len(self.scores)))
        for i, score in enumerate(self.scores):
            X_scores[:, i] = self.scores_clf[i].predict(X)
        X = np.concatenate((X, X_scores), axis=1)
        pred = self.target_clf.predict(X)
        return pred


class PredScoresThenClassifyV4(ClassifierMixin, BaseEstimator):

    def __init__(
        self,
        target="qcglobal",
        scores=SCORES,
        threshold=1.0,
        iqms=None,
        val_size=0.5,
        random_state=42,
    ):
        if target in scores:
            scores.remove(target)
        self.scores = scores
        self.target = target
        self.threshold = threshold
        self.scores_clf = [RandomForestRegressor() for _ in range(len(scores))]
        self.target_clf = RandomForestClassifier()
        self.iqms = iqms
        self.val_size = val_size
        self.random_state = random_state

    def fit(self, X, train_y):
        from sklearn.model_selection import train_test_split

        if self.iqms is not None:
            X = X[self.iqms]
        # Split X and train_y in two groups as 80 % for training and 20 % for testing
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, train_y, test_size=self.val_size, random_state=self.random_state
        )

        for i, score in enumerate(self.scores):
            tr_y_score = y_tr[score].astype(float)
            self.scores_clf[i].fit(X_tr, tr_y_score)

        pred_scores = np.zeros((X_te.shape[0], len(self.scores)))
        for i, score in enumerate(self.scores):
            pred_scores[:, i] = self.scores_clf[i].predict(X_te)
        X_te = np.concatenate((X_te, pred_scores), axis=1)
        self.target_clf.fit(X_te, y_te[self.target] >= self.threshold)
        return self

    def predict(self, X):
        if self.iqms is not None:
            X = X[self.iqms]
        # check_is_fitted(self)
        pred = np.zeros(X.shape[0])
        X_scores = np.zeros((X.shape[0], len(self.scores)))
        for i, score in enumerate(self.scores):
            X_scores[:, i] = self.scores_clf[i].predict(X)
        X = np.concatenate((X, X_scores), axis=1)
        pred = self.target_clf.predict(X)
        return pred


class RegressorClf(ClassifierMixin, BaseEstimator):
    def __init__(self, reg, threshold=1.0, rescale_y=False):
        self.reg = reg
        self.threshold = threshold
        self.rescale_y = rescale_y

    def fit(self, X, y):
        if self.rescale_y:
            # Do a sigmoid transformation of the target from 0-4 to 0-1, centered at threshold
            y = 1 / (1 + np.exp(-y + self.threshold))
        self.reg.fit(X, y)
        return self

    def predict(self, X):
        pred = self.reg.predict(X)
        if self.rescale_y:
            # Reverse the sigmoid transformation
            pred = -np.log(1 / pred - 1) + self.threshold
        pred_binary = pred >= self.threshold

        return pred_binary
from sklearn.neural_network import MLPClassifier

class ClfWrapper(ClassifierMixin, BaseEstimator):
    def __init__(self, clf, target, threshold=1.0, iqms=None):
        self.clf = clf
        self.target = target
        self.threshold = threshold
        self.iqms = iqms

    def fit(self, X, y, sample_weight=None):
        if self.iqms is not None:
            X = X[self.iqms]
        y = y[self.target] >= self.threshold
        if not isinstance(self.clf, MLPClassifier):
            self.clf.fit(X, y, sample_weight)
        else:
            self.clf.fit(X, y)
        return self

    def predict(self, X):
        if self.iqms is not None:
            X = X[self.iqms]
        return self.clf.predict(X)

    def predict_proba(self, X):
        if self.iqms is not None:
            X = X[self.iqms]
        return self.clf.predict_proba(X)


class RegWrapper(ClassifierMixin, BaseEstimator):

    def __init__(self, reg, target, iqms=None):
        self.reg = reg
        self.target = target
        self.iqms = iqms

    def fit(self, X, y):
        X_ = X.copy()
        if self.iqms is not None:
            X_ = X_[self.iqms]
        # Remove the target from the input data

        y = y[self.target]
        self.reg.fit(X_, y)
        return self

    def predict(self, X):
        X_ = X.copy()
        if self.iqms is not None:
            X_ = X_[self.iqms]
        return self.reg.predict(X_)


class StatsBasedPrediction(ClassifierMixin, BaseEstimator):
    def __init__(self, target="qcglobal", threshold=1.0):
        self.target = target
        self.threshold = threshold

    def fit(self, X, y, weight=None):
        res = np.mean(y[self.target] >= self.threshold)

        self.pred_proba = res
        self.pred = 0 if res < 0.5 else 1
        return self

    def predict(self, X):
        return np.array([self.pred] * X.shape[0])

    def predict_proba(self, X):
        return np.array([[1 - self.pred_proba, self.pred_proba]] * X.shape[0])

class StatsBasedRegressor(ClassifierMixin, BaseEstimator):
    def __init__(self, target="qcglobal"):
        self.target = target

    def fit(self, X, y):
        self.pred = np.mean(y[self.target])

        return self

    def predict(self, X):
        return np.array([self.pred] * X.shape[0])

class MultiLevelThresholdClf(ClassifierMixin, BaseEstimator):
    def __init__(self, target="qcglobal", threshold=1.0):
        self.target = target
        self.threshold = threshold
        self.thresholds = [1.0, 2.0, 3.0]
        self.clfs = [
            RandomForestClassifier() for _ in range(len(self.thresholds))
        ]

    def fit(self, X, y):
        for i, threshold in enumerate(self.thresholds):
            y_ = y[self.target] >= threshold
            self.clfs[i].fit(X, y_)
        return self

    def predict(self, X):
        preds = np.zeros((X.shape[0], len(self.thresholds)))
        for i, clf in enumerate(self.clfs):
            preds[:, i] = clf.predict(X)

        return np.sum(preds, axis=1) > sum(
            [t < self.threshold for t in self.thresholds]
        )
