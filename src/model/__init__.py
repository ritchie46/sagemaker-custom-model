from cloudhelper import open_s3_file
from wsgi import app
import pickle


class ModelWrap:
    def __init__(self):
        self._model = None
        self._scaler = None
        self._mean = None
        self._columns = None

    @property
    def model(self):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if self._model is None:
            f = open_s3_file(app.config['BUCKET'], app.config['MODEL_PKL'])
            self._model = pickle.load(f)
        return self._model

    @property
    def scaler(self):
        if self._scaler is None:
            f = open_s3_file(app.config['BUCKET'], app.config['SCALER_PKL'])
            self._scaler = pickle.load(f)
        return self._scaler

    @property
    def mean(self):
        if self._mean is None:
            f = open_s3_file(app.config['BUCKET'], app.config['MEAN_PKL'])
            self._mean = pickle.load(f)
        return self._mean

    @property
    def columns(self):
        if self._columns is None:
            f = open_s3_file(app.config['BUCKET'], app.config['COLUMNS_PKL'])
            self._columns = pickle.load(f)
        return self._columns

    def predict(self, x):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        return self.model.predict_proba(x)[:, 1]


modelwrapper = ModelWrap()