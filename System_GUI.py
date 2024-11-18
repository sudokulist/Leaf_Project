import os
import numpy as np
import random 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from PIL import Image, ImageTk
import torch
import torch.optim as optim
from torchvision import transforms
import tkinter as tk
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score
from Convolutional_AutoEncoder import Convolutional_AutoEncoder
from Classification import Classifier

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
train_test_rate = 0.2

nameAE = "Convolutional_AutoEncoder"
nameAE_pth = nameAE + '.pth'
nameAE_txt = nameAE + '.txt'
nameClass = "Classification"
nameClass_pth = nameClass + '.pth'
nameClass_txt = nameClass + '.txt'

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

AE_model1 = Convolutional_AutoEncoder().to(device)
AE_optimizer = optim.Adam(AE_model1.parameters(), lr = 0.001)
AE_model1.load_state_dict(torch.load(nameAE_pth, weights_only = True))
AE_model1.eval()
Class_model = Classifier().to(device)
Class_optimizer = optim.Adam(Class_model.parameters(), lr = 0.001)
Class_model.load_state_dict(torch.load(nameClass_pth, weights_only = True))
Class_model.eval()

def channel_similarity(image1, image2):
    similarities = []
    for i in range(image1.shape[0]):
        channel1 = image1[i, :, :] 
        channel2 = image2[i, :, :]  
        dot_product = torch.dot(channel1.view(-1), channel2.view(-1))  
        norm_product = torch.norm(channel1) * torch.norm(channel2)  
        similarity = dot_product / norm_product  
        similarity = similarity.cpu().detach().item()
        similarities.append(similarity)  
    return similarities

def dimension_getter(size1):
    dimension = []
    channel = 0
    width = 1
    height = 2
    dimension_channel = size1[channel]
    dimension_width = size1[width]
    dimension_height = size1[height]
    dimension.append(dimension_channel)
    dimension.append(dimension_width)
    dimension.append(dimension_height)
    return dimension

class ImageViewer:
    def __init__(self):
        self.width = new_width 
        self.height = new_height
        self.window = tk.Tk()
        self.window.title("NMirroring Pooling Image")
        self.flag = True
        self.strsize = 15
        self.label_map = label_map
        self.highlight_spot = (0, 0)
        self.label_pre = {}
        self.label_true = {}

        x, y, z, ans_label, similarity_image, input_dimension, enc_dimension, random_number = self.generate_images()
        if self.flag:
            Predicted_labels, True_labels, confusion, accuracy, precision, recall, f1 = self.Classification()
            self.Predicted_list = Predicted_labels
            self.True_list = True_labels
            self.Confusion = confusion
            self.flag = False

        self.AutoEncoder_label = tk.Label(self.window, text = "Convolutional AutoEncoder" , font = ("Times", self.strsize))
        self.AutoEncoder_label.grid(row = 0, column = 0, padx = 5, sticky = 'snew')
        compressrate = 1 - (enc_dimension[0] * enc_dimension[1] * enc_dimension[2]) / (input_dimension[0] * input_dimension[1] * input_dimension[2])
        self.compressibility_image_label = tk.Label(self.window, text = "Compressibility : " + '{:#04f}'.format(compressrate) , font = ("Times", self.strsize))
        self.compressibility_image_label.grid(row = 0, column = 1, columnspan = 3, sticky = 'w')

        self.canvas1 = tk.Canvas(self.window, width = self.width, height = self.height)
        self.canvas1.grid(row = 1, column = 0, rowspan = 6, padx = 5, pady = 5)
        self.canvas2 = tk.Canvas(self.window, width = self.width, height = self.height)
        self.canvas2.grid(row = 1, column = 1, rowspan = 6, padx = 5, pady = 5)
        self.canvas3 = tk.Canvas(self.window, width = self.width, height = self.height)
        self.canvas3.grid(row = 1, column = 2, rowspan = 6, padx = 5, pady = 5)
        
        self.image1_label = tk.Label(self.window, text = "Input Image : " + str(self.height) + " x " + str(self.width) ,font = ("Times", self.strsize))
        self.image1_label.grid(row = 7, column = 0, padx = 5, sticky = 'snew')
        self.image2_label = tk.Label(self.window, padx = 5, text = "Encoder Image : " + str(enc_dimension[1]) + " x " + str(enc_dimension[2]),
                                    font = ("Times", self.strsize))
        self.image2_label.grid(row = 7, column = 1, padx = 5, sticky = 'snew')
        self.image3_label = tk.Label(self.window, text = "Output image : " + str(self.height) + " x " + str(self.width), font = ("Times", self.strsize))
        self.image3_label.grid(row = 7, column = 2, padx = 5, sticky = 'snew')
        
        self.Compare_image = tk.Label(self.window, text = "Input Image to Output image Similarity", font = ("Times", self.strsize))
        self.Compare_image.grid(row = 1, column = 3, padx = 5, sticky = 'snew')
        self.red_similarity = tk.Label(self.window, text = "Red Similarity : " + '{:.4f}'.format(similarity_image[0]), font = ("Times", self.strsize), bd = 2, relief = "solid", bg = 'pink')
        self.red_similarity.grid(row = 2, column = 3, padx = 5, sticky = 'nsew')
        self.green_similarity = tk.Label(self.window, text = "Green Similarity : " + '{:.4f}'.format(similarity_image[1]), font = ("Times", self.strsize), bd = 2, relief = "solid", bg = 'lightgreen')
        self.green_similarity.grid(row = 3, column = 3, padx = 5, sticky = 'snew')
        self.blue_similarity = tk.Label(self.window, text = "Blue Similarity : " + '{:.4f}'.format(similarity_image[2]), font = ("Times", self.strsize), bd = 2, relief = "solid", bg = 'lightblue')
        self.blue_similarity.grid(row = 4, column = 3, padx = 5, sticky = 'nsew')
        Average_similarity = (similarity_image[0] + similarity_image[1] + similarity_image[2]) / 3.0
        self.avg_similarity = tk.Label(self.window, text = "Average Similarity : " + '{:.4f}'.format(Average_similarity), font = ("Times", self.strsize), bd = 2, relief = "solid")
        self.avg_similarity.grid(row = 5, column = 3, padx = 5, sticky = 'nsew')
        
        total_RedSimi, total_GreenSimi, total_BlueSimi = self.total_similarity()
        self.total_image = tk.Label(self.window, text = "Average Similarity across all Images", font = ("Times", self.strsize))
        self.total_image.grid(row = 1, column = 4, padx = 5, sticky = 'nsew')
        self.Total_RedSimi = tk.Label(self.window, text = "Avg Red Channel Similarity : " + '{:.4f}'.format(total_RedSimi), font = ("Times", self.strsize), bd = 2, relief = "solid", bg = 'pink')
        self.Total_RedSimi.grid(row = 2, column = 4, padx = 5, sticky = 'nsew')
        self.Total_GreenSimi = tk.Label(self.window, text = "Avg Green Channel Similarity : " + '{:.4f}'.format(total_GreenSimi), font = ("Times", self.strsize), bd = 2, relief = "solid", bg = 'lightgreen')
        self.Total_GreenSimi.grid(row = 3, column = 4, padx = 5, sticky = 'nsew')
        self.Total_BlueSimi = tk.Label(self.window, text = "Avg Blue Channel Similarity : " + '{:.4f}'.format(total_BlueSimi), font = ("Times", self.strsize), bd = 2, relief = "solid", bg = 'lightblue')
        self.Total_BlueSimi.grid(row = 4, column = 4, padx = 5, sticky = 'nsew')
        Average_Total_Similarity = (total_RedSimi + total_BlueSimi + total_GreenSimi) / 3.0
        self.Total_AveSimi = tk.Label(self.window, text = "Avg RGB Channel Similarity : " + '{:.4f}'.format(Average_Total_Similarity), font = ("Times", self.strsize), bd = 2, relief = "solid")
        self.Total_AveSimi.grid(row = 5, column = 4, padx = 5, sticky = 'nsew')
        
        self.highlight_spot = (self.True_list[random_number], self.Predicted_list[random_number])
        cmap = sns.color_palette("Blues", as_cmap = True)
        fig, ax = plt.subplots(figsize = (9, 6))
        sns.heatmap(self.Confusion, annot = True, fmt = 'd', cmap = cmap, ax = ax)
        ax.add_patch(plt.Rectangle((self.highlight_spot[1], self.highlight_spot[0]), 1, 1, fill = False, edgecolor = 'red', lw = 3))
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix with Highlight')
        canvas = FigureCanvasTkAgg(fig, master = self.window)
        canvas.get_tk_widget().grid(row = 9, column = 0, rowspan = 17, columnspan = 3, sticky = 'nswe')
        canvas.draw()
        
        self.Classification_label = tk.Label(self.window, text = "Image Classification Result", font = ("Times", self.strsize))
        self.Classification_label.grid(row = 8, column = 0, padx = 5, pady = 3, sticky = 'nswe')
        self.Accuracy = tk.Label(text = " Accuracy : " + '{:.4f}'.format(accuracy) + " , " + str(int(accuracy * 100)) + "%", font = ("Times", self.strsize), bd = 2, relief = "solid")
        self.Accuracy.grid(row = 9, column = 3, padx = 5, sticky = 'snew')
        self.Recall = tk.Label(self.window, text =  "Recall : " + '{:.4f}'.format(recall) + " , " + str(int(recall * 100)) + "%", font = ("Times", self.strsize), bd = 2, relief = "solid")
        self.Recall.grid(row = 10, column = 3, padx = 5, sticky = 'snew')
        self.Precision = tk.Label(text = " Precision : " + '{:.4f}'.format(precision) + " , " + str(int(precision * 100)) + "%", font = ("Times", self.strsize), bd = 2, relief = "solid")
        self.Precision.grid(row = 9, column = 4, padx = 5, sticky = 'snew')
        self.F1_Score = tk.Label(self.window, text = "F1 Score : " + '{:.4f}'.format(f1) + " , " + str(int(f1 * 100)) + "%", font = ("Times", self.strsize), bd = 2, relief = "solid")
        self.F1_Score.grid(row = 10, column = 4, padx = 5, sticky = 'snew')
        
        self.Pre_list_label = tk.Label(self.window, text = "Predicted_Label", font = ("Times", self.strsize), bd = 2, relief = "solid", bg = "pink")
        self.Pre_list_label.grid(row = 11, column = 3, sticky = 'nswe')
        self.True_list_label = tk.Label(self.window, text = "True_Label", font = ("Times", self.strsize), bd = 2, relief = "solid", bg = "lightblue")
        self.True_list_label.grid(row = 11, column = 4, sticky = 'nswe')
        rowi, Pre_col, True_col, number = 12, 3, 4, 0
        for i in range(class_number):
            self.label_pre[i] = tk.Label(self.window, text = str(number) + " : " + self.label_map[i], font = ("Times", self.strsize), bd = 2, relief = "solid")
            self.label_pre[i].grid(row = rowi, column = Pre_col, sticky = 'nswe')
            self.label_true[i] = tk.Label(self.window, text = str(number) + " : " + self.label_map[i], font = ("Times", self.strsize), bd = 2, relief = "solid")
            self.label_true[i].grid(row = rowi, column = True_col, sticky = 'nswe')
            rowi += 1
            number += 1
        
        self.button_exit = tk.Button(self.window, text = "Exit", width = 25, height = 2, command = self.window.quit, bg = "lightgreen")
        self.button_exit.grid(row = 27, column = 0, padx = 5, pady = 5)

        self.button_update = tk.Button(self.window, text = "Next Image", width = 25, height = 2, command = self.update_images, bg = "lightgreen")
        self.button_update.grid(row = 27, column = 1, padx = 5, pady = 5)

        self.run()
        
    def update_images(self):
        x, y, z, t, similarity_image, input_dimension, enc_dimension, random_number = self.generate_images()
        self.image1_array = Image.fromarray((x * 255).astype(np.uint8))
        self.image2_array = Image.fromarray((y * 255).astype(np.uint8))
        self.image3_array = Image.fromarray((z * 255).astype(np.uint8))

        self.image1_tk = ImageTk.PhotoImage(self.image1_array)
        self.image2_tk = ImageTk.PhotoImage(self.image2_array)
        self.image3_tk = ImageTk.PhotoImage(self.image3_array)

        self.canvas1.create_image(0, 0, anchor = tk.NW, image = self.image1_tk)
        self.canvas2.create_image(0, 0, anchor = tk.NW, image = self.image2_tk)
        self.canvas3.create_image(0, 0, anchor = tk.NW, image = self.image3_tk)
        
        self.red_similarity.config(text = "Red Similarity : " + '{:.4f}'.format(similarity_image[0]), font = ("Times", self.strsize))
        self.green_similarity.config(text = "Green Similarity : " + '{:.4f}'.format(similarity_image[1]), font = ("Times", self.strsize))
        self.blue_similarity.config(text = "Blue Similarity : " +  '{:.4f}'.format(similarity_image[2]), font = ("Times", self.strsize))
        Average_similarity = (similarity_image[0] + similarity_image[1] + similarity_image[2]) / 3.0
        self.avg_similarity.config(text = "Average Similarity : " + '{:.4f}'.format(Average_similarity), font = ("Times", self.strsize))
        
        old_high_spot = self.highlight_spot
        self.highlight_spot = (self.True_list[random_number], self.Predicted_list[random_number])
        plt.clf()
        plt.close()
        cmap = sns.color_palette("Blues", as_cmap = True)
        fig, ax = plt.subplots(figsize = (9, 6))
        sns.heatmap(self.Confusion, annot = True, fmt = 'd', cmap = cmap, ax = ax)
        ax.add_patch(plt.Rectangle((self.highlight_spot[1], self.highlight_spot[0]), 1, 1, fill = False, edgecolor = 'red', lw = 3))
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix with Highlight')
        canvas = FigureCanvasTkAgg(fig, master = self.window)
        canvas.get_tk_widget().grid(row = 9, column = 0, rowspan = 17, columnspan = 3, sticky = 'nswe')
        canvas.draw()
        
        self.label_pre[old_high_spot[1]].config(font = ("Times", self.strsize), bg = "white")
        self.label_true[old_high_spot[0]].config(font = ("Times", self.strsize), bg = "white")
        self.label_pre[self.highlight_spot[1]].config(font = ("Times", self.strsize, "bold"), bg = "pink")
        self.label_true[self.highlight_spot[0]].config(font = ("Times", self.strsize, "bold"), bg = "lightblue")
        
    def generate_images(self):
        random_number = random.randint(0, len(test_dataset))
        x, ans_label = test_dataset[random_number]
        x = x.to(device)
        z = x.unsqueeze(0)
        y = AE_model1.image_change(z)
        z = AE_model1(z)
        y = y.squeeze(0)
        z = z.squeeze(0)
        input_dimension = dimension_getter(x.size())
        enc_dimension = dimension_getter(AE_model1.encoder(x).size())
        similarity_image = channel_similarity(z, x)
        x = x.permute(1, 2, 0).cpu().numpy()
        y = y.permute(1, 2, 0).cpu().detach().numpy()
        z = z.permute(1, 2, 0).cpu().detach().numpy()
        return x, y, z, ans_label, similarity_image, input_dimension, enc_dimension, random_number
    
    def total_similarity(self):
        AE_model1.eval()
        Red_similarity = 0
        Green_similarity = 0
        Blue_similarity = 0
        for i in range(len(test_dataset)):
            image, ans_label = test_dataset[i]
            image = image.to(device)
            output_image = AE_model1(image.unsqueeze(0))
            output_image = output_image.squeeze(0)
            similarity = channel_similarity(image, output_image)
            Red_similarity += similarity[0]
            Green_similarity += similarity[1]
            Blue_similarity += similarity[2]
        return Red_similarity / len(test_dataset) , Green_similarity / len(test_dataset), Blue_similarity / len(test_dataset)
    
    def Classification(self):
        if self.flag:
            dataiter = iter(test_loader)
            Predicted_labels = []
            True_labels = []
            for data in dataiter:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    inputs = AE_model1.encoder(inputs)
                    outputs = Class_model(inputs)
                    predicted = torch.argmax(outputs, dim = 1)

                    for i in range(len(inputs)):
                        predicted_class_idx = predicted[i].item()
                        true_class_idx = labels[i].item()
                        Predicted_labels.append(predicted_class_idx)
                        True_labels.append(true_class_idx)

        accuracy = accuracy_score(True_labels, Predicted_labels)
        recall = recall_score(True_labels, Predicted_labels, average = 'macro')  
        precision = precision_score(True_labels, Predicted_labels, average = 'macro')   
        f1 = f1_score(True_labels, Predicted_labels, average = 'macro')
        confusion = confusion_matrix(True_labels, Predicted_labels)
        return Predicted_labels, True_labels, confusion, accuracy, precision, recall, f1

    def run(self):
        #self.window.configure(bg = 'cyan2') 
        self.window.mainloop()

viewer = ImageViewer()
viewer.run()