import datetime

from flask import Flask, request


def create_app():
    app = Flask(__name__)
    app.config.from_pyfile('settings.py')

    @app.post('/logs')
    def logs_add_post():
        data = request.json
        dataLogs = data.get('logs')
        for log in dataLogs:
            log['anomaly_score'] = 0.85
            log['result_timestamp'] = datetime.datetime.now()
        return dataLogs

    return app
