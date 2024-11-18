import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from Classification import Classifier
from Convolutional_AutoEncoder import Convolutional_AutoEncoder

np.random.seed(1)
torch.manual_seed(1)

if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 10
class_number = 14
new_width = 256
new_height = 256
epochs = 20
train_test_rate = 0.2

nameAE = "Classification"
nameAE_pth = nameAE + '.pth'
nameAE_txt = nameAE + '.txt'

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_image_paths_and_labels(data_dir):
    image_paths = []
    labels = []
    label_names = sorted(os.listdir(data_dir))  
    label_map = label_names  

    for label_idx, label_name in enumerate(label_names):
        label_dir = os.path.join(data_dir, label_name)
        if os.path.isdir(label_dir):
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                if img_path.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')):
                    image_paths.append(img_path)
                    labels.append(label_idx)  

    return image_paths, labels, label_map

def adjust_to_multiple_of_ten(paths, labels):
    size = len(paths)
    adjusted_size = size - (size % batch_size)
    return paths[:adjusted_size], labels[:adjusted_size]

def main(data_dir, test_size = train_test_rate, batch_size = batch_size):
    image_paths, labels, label_map = get_image_paths_and_labels(data_dir)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size = test_size, stratify = labels, random_state = 42
    )

    train_paths, train_labels = adjust_to_multiple_of_ten(train_paths, train_labels)
    test_paths, test_labels = adjust_to_multiple_of_ten(test_paths, test_labels)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((new_height, new_width), antialias = True),
    ])

    train_dataset = CustomDataset(train_paths, train_labels, transform = transform)
    test_dataset = CustomDataset(test_paths, test_labels, transform = transform)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, test_loader, test_dataset, label_map

data_dir = '14classes'
train_loader, test_loader, test_dataset, label_map = main(data_dir)

nameClass = "Classification"
nameClass_pth = nameClass + '.pth'
nameClass_txt = nameClass + '.txt'
AE_model = Convolutional_AutoEncoder().to(device)
AE_optimizer = optim.Adam(AE_model.parameters(), lr = 0.001)

with open(nameClass_txt, 'w') as file:
    Class_model = Classifier().to(device)
    Class_optimizer = optim.Adam(Class_model.parameters(), lr = 0.001)
    loss_fn = nn.CrossEntropyLoss() 
    
    print('Classifier')
    for epoch in range(epochs):
        print("Epoch: {}".format(epoch + 1))
        start_time = time.time()
        Class_model.train()
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            Class_optimizer.zero_grad()
            z = AE_model.encoder(inputs)
            outputs = Class_model(z)
            loss = loss_fn(outputs, labels)
            loss.backward()
            Class_optimizer.step()
        
        Class_model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                z = AE_model.encoder(inputs)
                outputs = Class_model(z)
                _, preds = torch.max(outputs, 1)
                loss = loss_fn(outputs, labels)
                train_loss += loss.item()
                total += inputs.size(0)
                correct += torch.sum(preds == labels.data)
            print("    Train Loss : {}".format(train_loss / (len(train_loader) / batch_size)) )
            print("    Train Accuracy : {}".format(100 * (correct / total)))
            train_accuracy = 100 * (correct / total)

        total = 0
        correct = 0
        with torch.no_grad():
            test_loss = 0.0
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                z = AE_model.encoder(inputs)
                outputs = Class_model(z)
                _, preds = torch.max(outputs, 1)
                loss = loss_fn(outputs, labels)
                test_loss += loss.item()
                total += inputs.size(0)
                correct += torch.sum(preds == labels.data)
    
            print("    Test Loss : {}".format(test_loss / (len(test_loader) / batch_size)))
            print("    Test Accuracy : {}".format(100 * (correct / total)))
            test_accuracy = 100 * (correct / total)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"    Execution time of the ClassifierModel: {execution_time:.4f} seconds")
        file.write(str(epoch + 1) + ',' + str(train_loss / (len(train_loader) / batch_size)) + "," + str(test_loss / (len(test_loader) / batch_size)) + "," + str(train_accuracy.item()) + ',' + str(test_accuracy.item()) + ',' + str(execution_time) + '\n')
torch.save(Class_model.state_dict(), nameClass_pth)