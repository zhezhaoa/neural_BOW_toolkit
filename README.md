# neural_BOW_toolkit

The code implements the neural bag-of-words models proposed in coling 2016: 
[Weighted Neural Bag-of-n-grams Model: New Baselines for Text Classification] (http://www.aclweb.org/anthology/C/C16/C16-1150.pdf)

**project**
We will upload the code and the datasets continually in the following weeks.
Our project include three components: (1) the datasets: nine binary classification datasets (2) liblinear: logistic regression (3) src: including entire source code

**datasets**
IMDB RT2k RTs subj AthR BbCrypt XGraph MPQA CR

One can download the datasets at [datasets](http://iir.ruc.edu.cn/~zhaoz/datasets.zip). Put the file in the path neural_BOW_toolkit/




**liblinear**

logistic regression

**source code**

The source code consists of three parts. (1) The dataset classes. Each .java file corresponds to a dataset, including the dataset's meta data and file loading operation.  (2) The model part. Including the entire training processing. (3) the utiliy parts. Sample class encapsulates the samples (or instances) in the datasets. LightSample class is a simplified version of Sample Class. LightSample class only stores information that is used during the training process. The Classifier class is used for calling the liblinear tool.   



**Acknowledgements**
Three projects help us a lot. The first is the work from Mesnil et al in 2015 ICLR workshop. They implement Paragraph Vector model in C at https://github.com/mesnilgr/iclr15. Another is the work done by Li et al. in 2016 ICLR workshop. They implement n-gram Paragraph Vector model in JAVA at https://github.com/libofang/DV-ngram. The last is the work from Wang and Manning. They propose the NBSVM and publish their code at https://github.com/sidaw/nbsvm. In fact, our models are the neural counterparts of NBSVM. We use the exactly the same datasets with NBSVM.
