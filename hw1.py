# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import os
import pandas as pd
import skimage
from skimage import io
from torch.utils.data import Dataset, DataLoader
import csv


class TrainCarsDataset(Dataset):
    def __init__(self, dir, csv_file, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.dir = dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Img read
        fname = '{:06d}'.format(self.annotations.iloc[index, 0])+'.jpg'
        img_path = os.path.join(self.dir, fname)
        image = io.imread(img_path)

        # Convert grayscale image to rgb image
        if len(image.shape) == 2:
            image = skimage.color.gray2rgb(image)

        car_type = self.annotations.iloc[index, 1]
        label = torch.tensor(classes.index(car_type))

        # Do transforms
        if self.transform:
            image = self.transform(image)

        return(image, label)


class TestCarsDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir
        self.transform = transform

        # Read all image from test data directory
        self.images = []
        for filename in os.listdir(dir):
            tmp = io.imread(os.path.join(dir, filename))
            if tmp is not None:
                self.images.append(tmp)

    def __len__(self):
        return 5000  # 5k test images

    def __getitem__(self, index):
        image = self.images[index].copy()

        # Convert grayscale image to rgb image
        if len(image.shape) == 2:
            image = skimage.color.gray2rgb(image)

        # Do transforms
        if self.transform:
            image = self.transform(image)

        return image


# Get the classes of cars
d = pd.read_csv('training_labels.csv', index_col=0,
                header=None, squeeze=True).to_dict()
classes = tuple(set(d.values()))

# For GPU usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
learning_rate = 0.001
batch_size = 32
num_epochs = 50

# Build Datasets & Loaders
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((450, 450)),
    transforms.RandomCrop((400, 400)),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomHorizontalFlip(p=0.25),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4493, 0.4303, 0.4319), (0.2902, 0.2875, 0.2937)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((450, 450)),
    transforms.Normalize((0.4493, 0.4303, 0.4319), (0.2902, 0.2875, 0.2937)),
])
train_set = TrainCarsDataset(
    'training_data/training_data', 'training_labels.csv', transform)
train_loader = DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True)
test_set = TestCarsDataset('testing_data/testing_data', transform_test)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size)
test_filenames = []
for filename in os.listdir('testing_data/testing_data'):
    tmp = filename.strip('0')[:-4]
    test_filenames.append(tmp)

# Model
model = models.googlenet(pretrained=True)
model.to(device)

# Loss,optimizer,scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, verbose=True)

# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent or Adam step
        optimizer.step()

    mean_loss = sum(losses)/len(losses)
    scheduler.step(mean_loss)
    print(f'Cost at epoch {epoch} is {mean_loss}')
print('---training done---')


# Generate My Prediction using test_loader
def gen_prediction(loader, model, predictions):
    model.eval()

    with torch.no_grad():
        for img in loader:
            img = img.to(device=device)
            scores = model(img)
            _, predict = scores.max(1)
            tmp_list = predict.tolist()
            predictions.extend(tmp_list)

    model.train()
    print('---predition done---')


predictions = []
gen_prediction(test_loader, model, predictions)

# Output the predictions to csv file
with open('hw1_prediction.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'label'])
    for i in range(5000):
        writer.writerow([test_filenames[i], classes[predictions[i]]])
