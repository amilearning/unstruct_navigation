import os


class TrainDir:
    def __init__(self):
        data_dir = os.path.expanduser('~') + '/data/'
        self.train_dir = os.path.join(data_dir, 'trainingData/')
        self.preprocessed_dir = os.path.join(data_dir, 'preprocessed/')
        self.eval_dir = os.path.join(data_dir, 'evaluationData/')
        self.model_dir = os.path.join(data_dir, 'models/')
        self.action_sample_dir = os.path.join(data_dir, 'actions/')
        self.snapshot_dir = os.path.join(data_dir, 'snapshot/')
        self.train_log_dir = os.path.join(data_dir, 'train_log/')
        self.image_dir = os.path.join(data_dir, 'img/')
        self.grid_dir = os.path.join(data_dir, 'grid/')

        self.normalizing_const_file = os.path.join(self.model_dir, 'normalizing_constant.pkl')