
# torch
import torch
import fastai
from fastai.vision.all import *
from fastai.vision.core import *

from Imageclassifier import *
from Configuration import *
from routing import *
# Flask utils

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath



#upload configurations
config = Config()
app = config.get_app()

#model uploads
learner = fastai.learner.load_learner("C:/Users/DELL/myproject/Models/deforestation_classifier.pkl")
state_dict = learner.model.state_dict()
model = torch.nn.Sequential(*list(learner.model.children()))
model.load_state_dict(state_dict)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = model.to(device)


def label_func(x):
    return mask_path+'/'+(x.stem + x.suffix)


model1 = fastai.learner.load_learner('C:/Users/DELL/myproject/Models/model9.pkl')



routing = Routing(app,model, model1)
routing.AppRouting()


if __name__ == "__main__":
    app.run(debug = True)