##################################################
# Exercise 2 - Natural Language Processing 67658  #
###################################################

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# subset of categories that we will use
category_dict = {'comp.graphics': 'computer graphics',
                 'rec.sport.baseball': 'baseball',
                 'sci.electronics': 'science, electronics',
                 'talk.politics.guns': 'politics, guns'
                 }


def get_data(categories=None, portion=1.):
    """
    Get data for given categories and portion.
    :param portion: portion of the data to use.
    """
    # get data
    from sklearn.datasets import fetch_20newsgroups
    data_train = fetch_20newsgroups(categories=categories, subset='train', remove=('headers', 'footers', 'quotes'),
                                    random_state=21)
    data_test = fetch_20newsgroups(categories=categories, subset='test', remove=('headers', 'footers', 'quotes'),
                                   random_state=21)

    # train
    train_len = int(portion * len(data_train.data))
    x_train = np.array(data_train.data[:train_len])
    y_train = data_train.target[:train_len]
    # remove empty entries
    non_empty = x_train != ""
    x_train, y_train = x_train[non_empty].tolist(), y_train[non_empty].tolist()

    # test
    x_test = np.array(data_test.data)
    y_test = data_test.target
    non_empty = np.array(x_test) != ""
    x_test, y_test = x_test[non_empty].tolist(), y_test[non_empty].tolist()
    return x_train, y_train, x_test, y_test

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Sub-task 1: TFIDF Vectorization
def vectorize_data(x_train, x_test):
    vectorizer = TfidfVectorizer(max_features=2000)
    x_train_tfidf = vectorizer.fit_transform(x_train).toarray()
    x_test_tfidf = vectorizer.transform(x_test).toarray()
    return x_train_tfidf, x_test_tfidf


# Sub-task 2: Convert Data to Tensors
def convert_to_tensors(x_train, y_train, x_test, y_test):
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor


# Sub-task 3: Define Model, Loss, and Optimizer
def define_model(input_dim, output_dim):
    nn = torch.nn
    optim = torch.optim
    model = nn.Sequential(
        nn.Linear(input_dim, output_dim)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, criterion, optimizer


def batch_train(criterion, model, optimizer, total_loss, x_batch, y_batch):
    optimizer.zero_grad()
    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    _, predicted = torch.max(outputs.data, 1)
    total_loss += loss.item()
    return total_loss


# Sub-task 4: Train Model
def train_model(model, criterion, optimizer, train_loader, test_loader, epochs=20):
    train_losses = []
    test_accuracies = []
    for epoch in range(epochs):
        model.train()

        total_loss = 0.0

        for x_batch, y_batch in train_loader:
            total_loss = batch_train(criterion, model, optimizer, total_loss, x_batch, y_batch)
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        test_accuracy = evaluate_model(model, test_loader)
        test_accuracies.append(test_accuracy)
        print(f"Log-Linear Classifier Accuracy: {test_accuracy:.4f}")
    return train_losses, test_accuracies


# Sub-task 5: Evaluate Model
def evaluate_model(model, test_loader):
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            outputs = model(x_batch)
            _, predicted = torch.max(outputs, 1)
            test_total += y_batch.size(0)
            test_correct += (predicted == y_batch).sum().item()
    test_accuracy = test_correct / test_total
    return test_accuracy


def plot_graphs(test_accuracies, portion, train_losses):
    # Plotting training loss and validation accuracy
    epochs = range(1, 21)
    plt.figure(figsize=(12, 5))
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training Loss for Portion {portion}')
    plt.xticks(np.arange(0, len(epochs), 1))
    plt.legend()

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Validation Accuracy for Portion {portion}')
    plt.xticks(np.arange(0, len(epochs), 1))
    plt.ylim(0.4, 1.0)
    plt.legend()

    plt.tight_layout()
    plt.show()


# Q1
def MLP_classification(portion=1.):
    """
    Perform linear classification
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    # Sub-task 1: Vectorize data
    x_train_tfidf, x_test_tfidf = vectorize_data(x_train, x_test)

    # Sub-task 2: Convert data to tensors
    x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = convert_to_tensors(x_train_tfidf, y_train,
                                                                                      x_test_tfidf, y_test)
    # Sub-task 3: Define model, loss, and optimizer
    input_dim = x_train_tfidf.shape[1]
    output_dim = len(category_dict)
    model, criterion, optimizer = define_model(input_dim, output_dim)

    # Create DataLoader for mini-batch training
    batch_size = 16
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Sub-task 4: Train model and evaluate
    train_losses, test_accuracies = train_model(model, criterion, optimizer, train_loader, test_loader)

    plot_graphs(test_accuracies, portion, train_losses)
    print(f"Log-linear model parameters: {count_trainable_parameters(model)}")
    return


# Q2
def MLP_hidden_layer_classification(portion=1.0):
    def define_mlp_model(input_dim, output_dim):
        model = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, output_dim)
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        return model, criterion, optimizer

    # Get the data
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    # Sub-task 1: Vectorize data
    x_train_tfidf, x_test_tfidf = vectorize_data(x_train, x_test)

    # Sub-task 2: Convert data to tensors
    x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = convert_to_tensors(x_train_tfidf, y_train,
                                                                                      x_test_tfidf, y_test)

    # Sub-task 3: Define MLP model with hidden layer, loss, and optimizer
    input_dim = x_train_tfidf.shape[1]
    output_dim = len(category_dict)
    model, criterion, optimizer = define_mlp_model(input_dim, output_dim)

    # Create DataLoader for mini-batch training
    batch_size = 16
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Sub-task 4: Train model and track loss
    train_losses, test_accuracies = train_model(model, criterion, optimizer, train_loader, test_loader)

    plot_graphs(test_accuracies, portion, train_losses)

    print(f"MLP with hidden layer parameters: {count_trainable_parameters(model)}")



# Q3
def transformer_classification(portion=1.):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from torch.utils.data import DataLoader
    import evaluate
    from tqdm import tqdm

    class Dataset(torch.utils.data.Dataset):
        """
        Dataset for loading data
        """

        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

        def __len__(self):
            return len(self.labels)

    def train_epoch(model, data_loader, optimizer, dev='cpu'):
        """
        Perform an epoch of training of the model with the optimizer
        :param model: Transformer model to be trained
        :param data_loader: DataLoader for training data
        :param optimizer: Optimizer (AdamW)
        :param dev: Device to use for training (CPU or GPU)
        :return: Average loss over the epoch
        """
        model.train()
        total_loss = 0.0
        # iterate over batches
        for batch in tqdm(data_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(dev)
            attention_mask = batch['attention_mask'].to(dev)
            labels = batch['labels'].to(dev)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            total_loss += loss.item()

        # Calculate the average loss for the epoch
        avg_loss = total_loss / len(data_loader)
        return avg_loss

    def evaluate_model(model, data_loader, dev='cpu', metric=None):
        """
        Evaluate the model on the validation set
        :param model: Transformer model to be evaluated
        :param data_loader: DataLoader for validation data
        :param dev: Device to use for evaluation (CPU or GPU)
        :param metric: Evaluation metric (e.g., accuracy)
        :return: Evaluation metric score
        """
        model.eval()
        for batch in tqdm(data_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(dev)
            attention_mask = batch['attention_mask'].to(dev)
            labels = batch['labels'].to(dev)

            # Forward pass (no gradient calculation needed)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            # Update metric with predictions and labels
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=labels)

        # Compute the final metric
        final_score = metric.compute()
        return final_score['accuracy']

    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    # Parameters
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_labels = len(category_dict)
    epochs = 3
    batch_size = 16
    learning_rate = 5e-5

    # Model, tokenizer, and metric
    model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=num_labels).to(dev)
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    metric = evaluate.load("accuracy")

    # Datasets and DataLoaders
    train_dataset = Dataset(tokenizer(x_train, truncation=True, padding=True), y_train)
    val_dataset = Dataset(tokenizer(x_test, truncation=True, padding=True), y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Add training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, dev=dev)
        print(f"Average Training Loss: {train_loss:.4f}")

        val_accuracy = evaluate_model(model, val_loader, dev=dev, metric=metric)
        print(f"Validation Accuracy: {val_accuracy:.4f}")

    print(f"Transformer model parameters: {count_trainable_parameters(model)}")

    return


if __name__ == "__main__":
    portions = [0.1, 0.2, 0.5, 1.]
    # Q1 - single layer MLP
    for portion in portions:
        print(f"Running log-linear classifier with portion {portion}")
        MLP_classification(portion=portion)

    # Q2 - multi-layer MLP
    for portion in portions:
        print(f"Running MLP classifier with hidden layer and portion {portion}")
        MLP_hidden_layer_classification(portion=portion)

    # Q3 - Transformer
    print("\nTransformer results:")
    for p in portions[:2]:
        print(f"Portion: {p}")
        transformer_classification(portion=p)
