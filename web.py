import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tkinter as tk
from tkinter import scrolledtext

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents file
data_file = open("intents.json").read()
intents = json.loads(data_file)

# Prepare data
words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Tokenize and process the data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the pattern
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Save words and classes for later use
pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))

# Prepare training data (bag of words and output tags)
training = []
output_empty = [0] * len(classes)

# Prepare the training set
for doc in documents:
    # Initialize our bag of words
    bag = []
    pattern_words = doc[0]  # tokenized pattern words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    # Create a bag of words array with 1 if word matches, else 0
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # Output row for the tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    # Append the bag of words and output to training data
    training.append([bag, output_row])

# Shuffle and convert into numpy arrays (make sure all inner lists have the same length)
random.shuffle(training)

# Separate the training data and labels
train_x = [item[0] for item in training]
train_y = [item[1] for item in training]

# Convert to numpy arrays
train_x = np.array(train_x)
train_y = np.array(train_y)

# Check that both arrays have the same length and shape
print("Train X shape:", train_x.shape)
print("Train Y shape:", train_y.shape)

# Define PyTorch Dataset
class ChatDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(self.Y[index], dtype=torch.float32)

# Create DataLoader
train_data = ChatDataset(train_x, train_y)
train_loader = DataLoader(train_data, batch_size=5, shuffle=True)

# Define the model using PyTorch
class ChatBotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatBotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = ChatBotModel(len(train_x[0]), 64, len(classes))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
epochs = 200
for epoch in range(epochs):
    for data in train_loader:
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, torch.max(labels, 1)[1])
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), 'model.pth')
print("Model trained and saved!")

# Chat function for Tkinter integration
def chat_response(user_input):
    # Tokenize the input
    message_words = nltk.word_tokenize(user_input)
    message_words = [lemmatizer.lemmatize(word.lower()) for word in message_words]
    
    # Create a bag of words
    bag = [0] * len(words)
    for w in message_words:
        if w in words:
            bag[words.index(w)] = 1
    
    # Convert to tensor
    input_tensor = torch.tensor(bag, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    # Predict
    output = model(input_tensor)
    
    # Ensure the output is of shape (batch_size, num_classes)
    if len(output.shape) == 1:
        output = output.unsqueeze(0)  # Add batch dimension if it's missing
    
    # Get the predicted class (tag)
    _, predicted = torch.max(output, dim=1)
    
    # Get the tag
    tag = classes[predicted.item()]
    
    # Find the response for the tag
    for intent in intents['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            break
    return response

# Tkinter GUI setup
def send_message():
    user_input = user_entry.get()
    if user_input.lower() == 'quit':
        root.quit()
    else:
        chat_window.config(state=tk.NORMAL)
        chat_window.insert(tk.END, "You: " + user_input + '\n')
        response = chat_response(user_input)
        chat_window.insert(tk.END, "Bot: " + response + '\n\n')
        chat_window.config(state=tk.DISABLED)
        user_entry.delete(0, tk.END)
        chat_window.yview(tk.END)
# ai_model.py

def get_ai_response(user_input):
    """
    A simple AI response function. Replace this with your actual AI model.
    """
    # Process the input and return a response
    response = f"AI says: {user_input}"  # Placeholder logic
    return response


# Set up the main Tkinter window
root = tk.Tk()
root.title("Chatbot")

# Set up chat window
chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=20, state=tk.DISABLED)
chat_window.grid(row=0, column=0, columnspan=2)

# Set up user input field
user_entry = tk.Entry(root, width=50)
user_entry.grid(row=1, column=0)

# Set up send button
send_button = tk.Button(root, text="Send", width=20, command=send_message)
send_button.grid(row=1, column=1)

# Run the Tkinter loop
root.mainloop()
