# celery_config.py

from celery import Celery

def make_celery(app):
    """
    Configures and returns a Celery instance that is integrated with a Flask app.
    This ensures that Celery tasks run within the Flask application context.
    """
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery