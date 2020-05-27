import argparse
import os
import time

import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import pickle,PIL
import numpy as np



torchvision.models.vgg.model_urls["vgg16"] = "http://webia.lip6.fr/~robert/cours/rdfia/vgg16-397923af.pth"
os.environ["TORCH_HOME"] = "/tmp/torch"
vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16.eval() 
imagenet_classes = pickle.load(open('imagenet_classes.pkl', 'rb'))
img = PIL.Image.open("dog.jpg")
img = img.resize((224, 224), PIL.Image.BILINEAR)
img = np.array(img, dtype=np.float32) / 255
img = img.transpose((2, 0, 1))

img = np.expand_dims(img, 0) # transformer en batch contenant une image
x = torch.Tensor(img)

y = vgg16.forward(x)
y = y.detach() # transformation en array numpy
y=y.numpy()
# TODO récupérer la classe prédite et son score de confiance
print(type(imagenet_classes))
print("la classe prédite est ",imagenet_classes[np.argmax(y)])
print('le score de confiance est',np.max(y))
print(vgg16.features[0](x).shape)

plt.imshow(vgg16.features[0](x)[0][0].detach())
plt.show()