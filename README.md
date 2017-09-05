# Deep-PPM: Deep Neural Framework for <b>P</b>redicting Point-of-Interest on Instagram Data
This page introduces our point-of-interest prediction framework. Our proposed model predcits <b>point-of-interest(POI)</b> on Instagram post utilizing user, text, and photo information in the post. Our proposed model embeds user information in dense user feature vectors and encodes text and photo information in textual and visual feature vectors, respectively, using the convolutional neural network. Not only does our proposed model not require the high-cost pre-processing stage, but it can also be learned on the much larger amount of datasets. 

## Model description
<p align="center">
<img src="/figures/model_description.jpg" width="400px" height="auto">
</p>
We want the user feature to capture a user’s POI preferences. We associate each user in the dataset with a real-valued embedding vector. A <b>user embedding matrix E</b> consists of the user embedding vector. We assume that certain objects will frequently appear in certain POIs. In doing so, we used the CNN model, which has been extensively studied in computer vision for effective visual feature extraction, as our <b>visual CNN layer</b>. We use all the layers of <a href="https://arxiv.org/pdf/1409.1556.pdf">VGGNet</a> except the softmax layer for our visual CNN layer. We want the textual feature to capture the textual context of a post. we use a <b>textual CNN layer</b> similar to the one presented in the work by <a href="http://www.aclweb.org/anthology/D14-1181">Kim et al.</a>.

## Data set
Data set is available at [here](https://s3.amazonaws.com/poiprediction/instagram.tar.gz). The data set includes "train.txt", "validation.txt", "test.txt", and "visual_feature.npz". The "train.txt"  "validation.txt" "test.txt" files include the training, validation, and tesing data respectively. The data is represented in the following format:
```bash
<post_id>\t<user_id>\t<word_1 word_2 ... >\t<poi_id>
```

All post_id, user_id, word_id, and poi_id are anonymized. Photo information also cannot be distributed due to personal privacy problems. So we relase the converted visual features from the output of the FC-7 layer of [VGGNet](https://arxiv.org/pdf/1409.1556.pdf) used as the visual feature extractor. If you want to use other visual feature extractor, such as [GoogleNet](http://arxiv.org/abs/1602.07261), [ResNet](https://arxiv.org/abs/1512.03385), you could implement it on your source code. We use a pre-trained VGGNet16 by [https://github.com/machrisaa/tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg) The "visual_feature.npz" file contains the visual features where the i-th row denotes i-th post's features.

### statistics
<table style="align=center;">
<tr><td>number of total post</td><td>number of POIs</td><td>number of users</td><td>size of vocabulary</td></tr>
<tr><td>736,445</td><td>9,745</td><td>14,830</td><td>470,374</td></tr>
<tr><td>size of training set</td><td>size of validation set</td><td>size of test set</td></tr>
<tr><td>526,783</td><td>67,834</td><td>141,828</td></tr>
</table>

## Getting Started
The code that implements our proposed model is implemented for the above dataset, which includes pre-processd visual feature. If you want to use a raw image that is not pre-processed, implement VGGNet on your source code as visual CNN layer.

### Prerequisites
- python 2.7
- tensorflow r0.12

### Usage
```bash
git clone https://github.com/heroiccharge202/poi-prediction
cd poi-prediction
python train_logistic_user_visual.py
```
