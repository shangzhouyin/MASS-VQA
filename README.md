# MASS-VQA
EC601-Group_Project

This is a implementation that is a variation of the Research Paper - Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering in PyTorch.

Visual feature are extracted using a pretrained (on ImageNet) ResNet-152. Input Questions are tokenized, embedded and encoded with an LSTM. Image features and encoded questions are combined and used to compute multiple attention maps over image features. The attended image features and the encoded questions are concatenated and finally fed to a 2-layer classifier that outputs probabilities over the answers (classes).

In order to consider all 10 answers given by the annotators we exploit a Soft Cross-Entropy loss : a weighted average of the negative log-probabilities of each unique ground-truth answer.

We also designed and implemented a GUI interface to show the training results.

Reference:https://github.com/DenisDsh/VizWiz-VQA-PyTorch
