import unittest
from wsgi import app
from model import ModelWrap
import cloudhelper


class InitTest(unittest.TestCase):

    def setUp(self):
        app.config['DEBUG'] = False
        app.config['TESTING'] = True
        self.app = app.test_client()
        self.modelwrapper = ModelWrap()


class MainTests(InitTest):

    def test_load_model_from_s3(self):
        self.assertIn('sklearn', str(type(self.modelwrapper.model)))

    def test_load_mean_values_from_s3(self):
        self.assertTrue(self.modelwrapper.mean.dtype == float)

    def test_load_scaler_from_s3(self):
        self.assertIn('scaler', self.modelwrapper.scaler.__str__().lower())

    def test_load_columns_from_s3(self):
        self.assertTrue(len(self.modelwrapper.columns) > 0)

    def test_model(self):
        f = cloudhelper.open_s3_file(app.config['BUCKET'], app.config['TEST_DATA'])

        response = self.app.post('/invocations', data=f.read().decode('utf-8'), content_type='text/csv')
        self.assertEqual(response.status_code, 200)

