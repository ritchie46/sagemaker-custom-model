import os
os.environ['MODEL_PATH'] = os.path.dirname(__file__)


if __name__ == '__main__':
    from container.model.predictor import app
    app.run(port=8080)
