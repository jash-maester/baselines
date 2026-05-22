"""LDCQ hard-codes `Experiment(api_key='', project_name='')` which crashes the
real comet_ml on empty credentials. This stub makes it a no-op."""


class Experiment:
    def __init__(self, *args, **kwargs):
        pass

    def log_parameters(self, params):
        pass

    def log_metric(self, key, value, step=None):
        pass

    def log_model(self, *args, **kwargs):
        pass

    def add_tag(self, tag):
        pass

    def end(self):
        pass

    def __getattr__(self, name):
        return lambda *a, **kw: None


class ExistingExperiment(Experiment):
    pass


class OfflineExperiment(Experiment):
    pass


def login(*args, **kwargs):
    return True
