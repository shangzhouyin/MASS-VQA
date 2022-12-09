# MASS-VQA
EC601-Group_Project

This is a implementation that is a variation of the Research Paper - Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering in PyTorch.

Visual feature are extracted using a pretrained (on ImageNet) ResNet-152. Input Questions are tokenized, embedded and encoded with an LSTM. Image features and encoded questions are combined and used to compute multiple attention maps over image features. The attended image features and the encoded questions are concatenated and finally fed to a 2-layer classifier that outputs probabilities over the answers (classes).

In order to consider all 10 answers given by the annotators we exploit a Soft Cross-Entropy loss : a weighted average of the negative log-probabilities of each unique ground-truth answer.

![image](https://user-images.githubusercontent.com/113374250/206721018-caebc9a8-96c4-44f8-a284-8f8726b3d345.png)


We also designed and implemented a GUI interface to show the training results.
![image](https://user-images.githubusercontent.com/113374250/206720901-013bcbc8-f3f2-4f9b-b805-6a2220ccbc8e.png)

Reference:https://github.com/DenisDsh/VizWiz-VQA-PyTorch
