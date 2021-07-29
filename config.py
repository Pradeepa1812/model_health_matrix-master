import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Base():
    DEBUG = False
    # Your App secret key
    #SECRET_KEY = "\2\1thisismyscretkey\1\2\e\y\y\h"

    API_KEY = "f0e228cf4535857f2482c1c433eb95d69ba664aab8fc010c523fbaf47921779e"

    #SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Flask-WTF flag for CSRF
    CSRF_ENABLED = True