# Image Captioning:
* The Microsoft Common Objects in COntext (MS COCO) dataset is a large-scale dataset for scene understanding. 
* The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms.
* More about the dataset on the website or in this research paper (https://arxiv.org/pdf/1405.0312.pdf).
## ``data_loader.py``
* The data_loader is loaded in Notebook 2 by calling the get_loader function defined in this file
* We can access the corresponding dataset as data_loader.dataset. This dataset is an instance of the CoCoDataset class in data_loader.py.
* The ``__getitem__`` method in the CoCoDataset class determines how an image-caption pair is pre-processed before being incorporated into a batch
*  data_loader.dataset.vocab is an instance of the Vocabulary class from vocabulary.py, it is used to pre-process the captions, converting any string-valued caption to a list of integers
## ``vocabulary.py``
* The __init__ builds up the vocabulary
*  The word2idx dictionary is created in this file by looping over the captions in the training dataset

## ``model.py``
* The model is defined in this file.
* There are two classes EncoderCNN and DecoderCNN defined in this file.
* The EncoderCNN convert an image t
* The encoder is a pre-trained ResNet-50 architecture (with the final fully-connected layer removed) to extract features from a batch of pre-processed images. The output is then flattened to a vector, before being passed through a Linear layer to transform the feature vector to have the same size as the word embedding.
* The DecoderRNN class accept's as input, features containing the embedded image features along with a PyTorch tensor corresponding to the last batch of captions.
