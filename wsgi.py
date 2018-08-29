import flask

app = flask.Flask(__name__)
app.config.from_pyfile('./instance/env.cfg')

from web.router import app