#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""docstring
"""

import os
import sys
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import numpy as np
import seg_reader

parrots_home = os.environ.get('PARROTS_HOME')
sys.path.append(os.path.join(parrots_home, 'parrots', 'python'))

# from pyparrots.dnn import Session, Model, config
from parrots.dnn import Session, Model, config

def model_and_session(model_file, session_file):
    with open(model_file) as fin:
        model_text = fin.read()
    with open(session_file) as fin:
        session_cfg = yaml.load(fin.read(), Loader=Loader)
    session_cfg = config.ConfigDict(session_cfg)
    session_cfg = config.ConfigDict.to_dict(session_cfg)
    session_text = yaml.dump(session_cfg, Dumper=Dumper)

    model = Model.from_yaml_text(model_text)
    session = Session.from_yaml_text(model, session_text)

    return model, session

def predict(inputs):
    model_file = './work_dir/model.yaml'
    session_file = './work_dir/session.yaml'
    param_file = 'work_dir/snapshots/iter.latest.parrots'

    model, session = model_and_session(model_file, session_file)

    flow = session.flow('val')
    flow.load_param(param_file)
 
    img = inputs['data'].astype(dtype=np.float32, order='F')
    img = np.reshape(img, 480, 480, 3, 2)
    
    flow.set_input('data', img)
    flow.forward()
    pred = flow.data('upscore2').value()
    return pred.T
    
def main():
    # model_file = 'model.yaml'
    model_file = './work_dir/model.yaml'
    session_file = './work_dir/session.yaml'
    # param_file = 'iter.latest.parrots'
    param_file = 'work_dir/snapshots/iter.latest.parrots'

    model, session = model_and_session(model_file, session_file)
    session.setup()

    flow = session.flow('train')
    flow.load_param(param_file)

    img = np.zeros((480, 480, 3, 2), dtype=np.float32, order='F')

    print img.shape
    flow.set_input('data', img)
    flow.forward()
    pred = flow.data('upscore2').value()
    print flow.data('label').value().shape
    print flow.data('label_weight').value().shape
    print flow.data('upscore2').value().shape


    # with session.flow('train') as flow:
    #     flow.init_param()
    #     for i in range(100):
    #         flow.feed()


if __name__ == '__main__':
    main()

