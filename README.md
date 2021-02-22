# Main idea about the content

This repository contains in total 4 Homeworks, which each one is part of the *Artificial Intelligence II (Deep Learning forNatural Language Processing) Fall Semester 2020, Winter of 2020, University of Athens, DiT.*
Each Homework contains a specific .pdf file which explains the homework's requirements. For each Homework also, we have multiple well-document .ipynb files.

# HW1: :heavy_check_mark:

This Homework was an introduction about Regression. More specifically, from this homework we've taught about:

1. [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent)
2. [Stohastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
3. [Mini-Batch Gradient Descent](https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/)

We were asked to implement the above gradient descent methods for **Ridge Regression**, bit much from *the scratch.*

Also, we used the sklearn's [Logistic Regression's Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), in order to do some sentiment analysis on [this](https://drive.google.com/file/d/1dTIWNpjlrnTQBIQtaGOh0jCRYZiAQO79/view) Twitter's dataset.

# HW2: :heavy_check_mark:

This specific Homework, was an brave introduction about Neural Networks, and more specifically about [FeedForward Neural Networks](https://en.wikipedia.org/wiki/Feedforward_neural_network).

We have configured some hyperparameters in our final model, such as:

1. The number of hidden layers and the number of their units
2. The activation functions
3. The loss functions
4. The optimizer

# HW3: :heavy_check_mark:

For this particular Homework , we were asked to develop a sentiment classifier using a **Bidirectional stacked RNN with LSTM/GRU cells**.

We also worked with pre-trained word embeddings [GloVe](https://nlp.stanford.edu/projects/glove/).

# HW4: :heavy_check_mark:

For the last homework about this course, we developed a **document retrieval system to return titles of scientific papers containing the answer to a given user question!**

In order to do this, we've have utilized the power of [SBERT](https://github.com/UKPLab/sentence-transformers).

We have developed 2 models. Each model has a different sentence embedding approach. 

1. bert-base-nli-mean-tokens 
2. msmarco-distilbert-base-v2 

For the 2nd approach, we also used the cross-encoder/ms-marco-TinyBERT-L-6 model, in order to fiilter our certain noise that might be retrieved from the semantic search step.

