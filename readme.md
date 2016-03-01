#Image classification

##ImageNet

### Googlenet
This is an implementation of the GoogLeNet model for image classification described in [Szegedy et. al. 2014](http://arxiv.org/pdf/1409.4842.pdf).

Model script and weights can be found at [googlenet_neon](https://gist.github.com/nervanazoo/2e5be01095e935e90dd8).

Citation:
```
Going deeper with convolutions
Szegedy, Christian; Liu, Wei; Jia, Yangqing; Sermanet, Pierre;
Reed, Scott; Anguelov, Dragomir; Erhan, Dumitru; Vanhoucke, Vincent;
Rabinovich, Andrew
arXiv:1409.4842
```

###VGG

We have adapted the 16 and 19 layer VGG model that is available for Caffe for use with neon.  

Model script and weights can be found at [neon VGG](https://gist.github.com/nervanazoo/e74ebe6418852f547aa8).

Citation:

```
Very Deep Convolutional Networks for Large-Scale Image Recognition
K. Simonyan, A. Zisserman
arXiv:1409.1556
```

###Alexnet

An implementation of an image classification model based on [Krizhevsky, Sutskever and Hinton 2012](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).

Model script and weights can be found at [Alexnet neon](https://gist.github.com/nervanazoo/14bb75d2bb5f20d9c482).


Citation:
```
ImageNet Classification with Deep Convolutional Neural Networks
Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton
Advances in Neural Information Processing Systems 25
eds.F. Pereira, C.J.C. Burges, L. Bottou and K.Q. Weinberger
pp. 1097-1105, 2012
```

##Cifar10

###Deep Residual Network
An implementation of deep residual networks as described in [He, Zhang, Ren, Sun 2015](http://arxiv.org/abs/1512.03385).

Model script and weights can be found at [cifar10_msra neon](https://github.com/apark263/cfmz).

Citation:
```
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
Deep Residual Learning for Image Recognition
arXiv preprint arXiv:1512.03385, 2015.
```

###AllCNN
An implementation of a deep convolutional neural network model inspired by the paper [Springenberg, Dosovitskiy, Brox, Riedmiller 2014](http://arxiv.org/abs/1412.6806). 

Model script and weights can be found at [cifar10_allcnn neon](https://gist.github.com/nervanazoo/47198f475260e77f64fe).

Citation:
```
Jost Tobias Springenberg,  Alexey Dosovitskiy, Thomas Brox and Martin A. Riedmiller. 
Striving for Simplicity: The All Convolutional Net. 
arXiv preprint arXiv:1412.6806, 2014.
```

#Object localization

##Fast-RCNN
Fast-RCNN model trained on PASCAL VOC dataset. The CNN layers are seeded by a [Alexnet pre-trained in neon](https://gist.github.com/nervetumer/64fe5ea27569c9042d8b) using ImageI1K dataset.

Model, weights, and instructions can be found at [here](https://gist.github.com/yinyinl/d12a82dc11df79067740).

Citation:
```
Fast R-CNN
http://arxiv.org/pdf/1504.08083v2.pdf
```
```
https://github.com/rbgirshick/fast-rcnn
```

#Scene Classification

##Deep ResNet on (mini-)Places2

Implementation of a [deep residual network](http://arxiv.org/abs/1512.03385) applied to Mini-Places2.

Model, weights, and instructions can be found at [here](https://github.com/hunterlang/mpmz/).


#Image Captioning
LSTM image captioning model based on [CVPR 2015 paper](http://arxiv.org/abs/1411.4555): 

    Show and tell: A neural image caption generator.
    O. Vinyals, A. Toshev, S. Bengio, and D. Erhan.  
    CVPR, 2015 (arXiv ref. cs1411.4555)

and code from Karpathy's [NeuralTalk](https://github.com/karpathy/neuraltalk).

Models:
* [LSTM-flickr8k](https://gist.github.com/nervanazoo/9b276eaee644d723f4b6): Single layer 512 hidden unit LSTM trained on flickr8k dataset.

#Deep reinforcement learning

An implementation of a deep Q-network that can learn to play video games based on [Mnih et. al. Nature 2015](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html).

This was developed by Tambet Matiisen at University of Tartu in Estonia.  Code and instructions for use can be found at [simple_dqn](https://github.com/tambetm/simple_dqn).  Trained model snapshots are provided in the repo as well ([link to snapshots](https://github.com/tambetm/simple_dqn/tree/master/snapshots)).

Citation:
```
V. Mnih et. al.
Human-level control through deep reinforcement learning
Nature 518, 529–533 (26 February 2015)
```

#NLP: Question Answering
##bAbI
Facebook's baseline GRU/LSTM model on the bAbI dataset.

Model, weights, and instructions can be found [here](https://gist.github.com/nervanazoo/3277f9fafd429cb41081).


[Facebook research link](https://research.facebook.com/researchers/1543934539189348)

```
Weston, Jason, et al. "Towards AI-complete question answering: a set of prerequisite toy tasks." arXiv preprint arXiv:1502.05698 (2015).
```

#NLP: Sentiment classification
##IMDB
LSTM model for solving the IMDB sentiment classification task.

Model, weights, and instructions can be found [here](https://gist.github.com/nervanazoo/976ec931bb4549131ae0).

```
When Are Tree Structures Necessary for Deep Learning of Representations?
Jiwei Li, Dan Jurafsky and Eduard Hovy
http://arxiv.org/pdf/1503.00185v1.pdf
```

#Video
##C3D
C3D model trained on UCF101 dataset.

Model, weights, and instructions can be found [here](https://gist.github.com/SNagappan/304446c6c2f7afe29629).

```
Learning Spatiotemporal Features with 3D Convolutional Networks
http://arxiv.org/pdf/1412.0767v4.pdf
```
```
http://vlg.cs.dartmouth.edu/c3d/
```
```
https://github.com/facebook/C3D
```
