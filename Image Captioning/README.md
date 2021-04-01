# Image Captioning:
* The Microsoft Common Objects in COntext (MS COCO) dataset is a large-scale dataset for scene understanding. 
* The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms.
* More about the dataset on the website or in this research paper (https://arxiv.org/pdf/1405.0312.pdf).
### ``data_loader.py``
* The data_loader is loaded in Notebook 2 by calling the get_loader function defined in this file
* We can access the corresponding dataset as data_loader.dataset. This dataset is an instance of the CoCoDataset class in data_loader.py.
* The ``__getitem__`` method in the CoCoDataset class determines how an image-caption pair is pre-processed before being incorporated into a batch
*  data_loader.dataset.vocab is an instance of the Vocabulary class from vocabulary.py, it is used to pre-process the captions, converting any string-valued caption to a list of integers
### ``vocabulary.py``
* The __init__ builds up the vocabulary
*  The word2idx dictionary is created in this file by looping over the captions in the training dataset

### ``model.py``
* The model is defined in this file.
* There are two classes EncoderCNN and DecoderCNN defined in this file.
* The EncoderCNN convert an image t
* The encoder is a pre-trained ResNet-50 architecture (with the final fully-connected layer removed) to extract features from a batch of pre-processed images. The output is then flattened to a vector, before being passed through a Linear layer to transform the feature vector to have the same size as the word embedding.
* The DecoderRNN class accept's as input, features containing the embedded image features along with a PyTorch tensor corresponding to the last batch of captions.
* Output on passing feature and captions to decoderis a PyTorch tensor with size ``[batch_size, captions.shape[1], vocab_size]`` such that ``outputs[i,j,k]`` contains the model's predicted score, indicating how likely the j-th token in the i-th caption in the batch is the k-th token in the vocabulary.
* Sample method in the DecoderRNN class accept's as input embedded input features corresponding to a single image.
* It return's as output a Python list output, indicating the predicted sentence. output[i] is a nonnegative integer that identifies the predicted i-th token in the sentence. 
* The correspondence between integers and tokens is in the wor2idx dictionary.

### ``Notebook 3``
* The training of the model is done in this notebook
* We inititialize the data_loader, encoder and decoder. Only the decoder parametrs and encoder last embed layer parameters are trainable.


## Sample Results
______________
![alt text](https://github.com/varangrai/Udacity-CVND-Projects/blob/master/Image%20Captioning/Results.png?raw=true)
