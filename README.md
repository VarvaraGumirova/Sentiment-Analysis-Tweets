Project Description: Sentiment Analysis Model with PyTorch

This project implements a sentiment analysis model using PyTorch to classify tweets as positive or negative. The model leverages text preprocessing techniques, including lowercasing and removing non-alphabetic characters, to clean the data. To handle class imbalance, undersampling is applied to balance the dataset. The neural network architecture includes an embedding layer, a hidden layer with ReLU activation, and dropout for regularization. The model is trained using BCEWithLogitsLoss and optimized with Adam.

Key Features:

Text Preprocessing: Converts text to lowercase and removes punctuation.
Undersampling: Balances the dataset by reducing the number of negative samples.
Neural Network Architecture: Uses an embedding layer, a hidden layer with ReLU activation, and dropout for regularization.
Training: Trains the model on a balanced dataset of 100,000 tweets.
Testing: Demonstrates the model's performance on a sample tweet.
The next step tp test the model on the test data is to be done.

Possible Improvements:
Add cross-validation for better evaluation.
Experiment with different architectures (e.g., LSTM, GRU).
Use pre-trained embeddings like GloVe or Word2Vec.
