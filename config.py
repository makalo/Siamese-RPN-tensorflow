import os
import numpy as np

class Config:
    def __init__(self):
        self._configs = {}

        # Training Config
        self._configs["root_dir"]           = './data/VID_ALL'
        self._configs["model_dir"]          = './checkpoint/VID_ALL'
        self._configs["pre_trained_dir"]    = './pretrained'
        self._configs["learning_rate"]      = 0.01
        self._configs["decay_rate"]         = 0.95
        self._configs["decay_step"]         = 5000
        self._configs["epoch_num"]          = 50
        self._configs["snapshot_name"]      = 'SiameseRPN'

        # Test Config
        self._configs["img_path"]           = './data/VID/ILSVRC2015_train_00004000'
        self._configs["label_path"]         = './data/VID/ILSVRC2015_train_00004000/groundtruth.txt'
        self._configs["penalty_k"]          = 0.055
        self._configs["window_influence"]   = 0.42
        self._configs["lr"]                 = 0.12
        # Vedio Config
        self._configs["vedio_dir"]          = './data/vedio'
        self._configs["vedio_name"]         = 'test.mp4'
        #Debug
        self._configs["debug_dir"]          = "./debug"
    @property
    def root_dir(self):
        return self._configs["root_dir"]
    @property
    def model_dir(self):
        model_dir=self._configs["model_dir"]
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir
    @property
    def pre_trained_dir(self):
        pre_trained_dir=self._configs["pre_trained_dir"]
        if not os.path.exists(pre_trained_dir):
            os.makedirs(pre_trained_dir)
        return pre_trained_dir
    @property
    def learning_rate(self):
        return self._configs["learning_rate"]
    @property
    def decay_rate(self):
        return self._configs["decay_rate"]
    @property
    def decay_step(self):
        return self._configs["decay_step"]
    @property
    def epoch_num(self):
        return self._configs["epoch_num"]
    @property
    def snapshot_name(self):
        return self._configs["snapshot_name"]
    @property
    def img_path(self):
        return self._configs["img_path"]
    @property
    def label_path(self):
        return self._configs["label_path"]
    @property
    def penalty_k(self):
        return self._configs["penalty_k"]
    @property
    def window_influence(self):
        return self._configs["window_influence"]
    @property
    def lr(self):
        return self._configs["lr"]
    @property
    def vedio_dir(self):
        vedio_dir=self._configs["vedio_dir"]
        if not os.path.exists(vedio_dir):
            os.makedirs(vedio_dir)
        return vedio_dir
    @property
    def vedio_name(self):
        return self._configs["vedio_name"]
    @property
    def debug_dir(self):
        debug_dir = self._configs["debug_dir"]
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        return debug_dir
    def update_config(self, new):
        for key in new:
            if key in self._configs:
                self._configs[key] = new[key]

cfg = Config()
