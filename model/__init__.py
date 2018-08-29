from cloudhelper import open_s3_file
from wsgi import app
import pickle


class ModelWrap:
    model = None
    scaler = None
    mean = None
    columns = None

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            f = open_s3_file(app.config['BUCKET'], app.config['MODEL_PKL'])
            cls.model = pickle.load(f)
        return cls.model

    @classmethod
    def get_scaler(cls):
        if cls.scaler is None:
            f = open_s3_file(app.config['BUCKET'], app.config['SCALER_PKL'])
            cls.scaler = pickle.load(f)

    @classmethod
    def get_mean_values(cls):
        if cls.mean is None:
            f = open_s3_file(app.config['BUCKET'], app.config['MEAN_PKL'])
            cls.mean = pickle.load(f)

    @classmethod
    def get_column_values(cls):
        if cls.mean is None:
            f = open_s3_file(app.config['BUCKET'], app.config['COLUMNS_PKL'])
            cls.columns = pickle.load(f)

    @classmethod
    def predict(cls, x):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()
        return clf.predict_proba(x)[:, 1]
