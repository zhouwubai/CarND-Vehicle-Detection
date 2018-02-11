from enum import Enum
from sklearn import svm
from sklearn import tree
from sklearn import naive_bayes
from sklearn.preprocessing import StandardScaler

from car.features import extract_features
from car.search import fast_search_windows


class ModelType(Enum):
    SVC = 'SVC'
    LinearSVC = 'LinearSVC'
    GuassianNB = 'GuassianNB'
    DecisionTree = 'DecisionTree'


def get_classifier(name=ModelType.SVC, C=1.0, kernel='rbf'):
    if name == ModelType.SVC:
        return svm.SVC(C=C, kernel=kernel)
    elif name == ModelType.LinearSVC:
        return svm.LinearSVC(C=C)
    elif name == ModelType.GuassianNB:
        return naive_bayes.GaussianNB()
    elif name == ModelType.DecisionTree:
        return tree.DecisionTreeClassifier()
    else:
        raise NotImplementedError


class CarDetector(object):
    """
    This class wraps parameter and methods for detect car
    """
    def __init__(self, model_type=ModelType.SVC, C=1.0, kernel='rbf',
                 color_space='HLS',
                 spatial_feat=True, spatial_size=(16, 16),
                 hist_feat=True, hist_bins=16,
                 hog_feat=True, orient=16,
                 pix_per_cell=16, cell_per_block=2, hog_channel='ALL'):

        self.model_type = model_type
        self.model = get_classifier(model_type, C, kernel)
        self.color_space = color_space
        self.spatial_feat = spatial_feat
        self.spatial_size = spatial_size
        self.hist_feat = hist_feat
        self.hist_bins = hist_bins
        self.hog_feat = hog_feat
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.X_scaler = StandardScaler()
        self.normalize = False

    def train(self, X, y, normalize=True):
        self.normalize = normalize
        if self.normalize:
            self.X_scaler.fit(X)
            scaled_X = self.X_scaler.transform(X)
        else:
            scaled_X = X

        self.model.fit(scaled_X, y)

    def evaluate(self, X, y):
        if self.normalize:
            scaled_X = self.X_scaler.transform(X)
        else:
            scaled_X = X
        score = self.model.score(scaled_X, y)
        return score

    def get_features(self, files):
        return extract_features(files, color_space=self.color_space,
                                spatial_size=self.spatial_size,
                                hist_bins=self.hist_bins,
                                orient=self.orient,
                                pix_per_cell=self.pix_per_cell,
                                cell_per_block=self.cell_per_block,
                                hog_channel=self.hog_channel,
                                spatial_feat=self.spatial_feat,
                                hist_feat=self.hist_feat,
                                hog_feat=self.hog_feat)

    def search_windows(self, img, y_start_stop,
                       scale, cells_per_step):
        if not isinstance(y_start_stop, list):
            y_start_stop = [y_start_stop]
        if not isinstance(scale, list):
            scale = [scale]
        if not isinstance(cells_per_step, list):
            cells_per_step = [cells_per_step]

        min_len = min(len(y_start_stop), len(scale), len(cells_per_step))
        y_start_stop = y_start_stop[:min_len]
        scale = scale[:min_len]
        cells_per_step = cells_per_step[:min_len]

        on_windows = []
        for y_ss, s, c in zip(y_start_stop, scale, cells_per_step):
            windows = fast_search_windows(
                img=img, clf=self.model, X_scaler=self.X_scaler,
                ystart=y_ss[0], ystop=y_ss[1], scale=s,
                color_space=self.color_space,
                spatial_size=self.spatial_size,
                hist_bins=self.hist_bins,
                orient=self.orient,
                pix_per_cell=self.pix_per_cell,
                cell_per_block=self.cell_per_block,
                cells_per_step=c
            )
            on_windows.extend(windows)
        return on_windows

    def print_params(self):
        param = 'Model: {}, Color space: {}\n'.format(self.model_type,
                                                      self.color_space)
        if self.spatial_feat:
            param += 'Using spatial features\n' +\
                     'spatial_size={}\n'.format(self.spatial_size)
        if self.hist_feat:
            param += 'Using histogram features\n' +\
                     'hist_bins={}\n'.format(self.hist_bins)
        if self.hog_feat:
            param += 'Using hog features\n' +\
                     'orient={}\n'.format(self.orient) +\
                     'pix_per_cell={}\n'.format(self.pix_per_cell) +\
                     'cell_per_block={}\n'.format(self.cell_per_block) +\
                     'hog_channel={}\n'.format(self.hog_channel)
        return param


