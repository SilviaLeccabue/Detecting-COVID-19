%matplotlib inline

import os
import shutil
import random
import torch
import torchvision
import numpy as np
from torchvision.transforms.functional import normalize, resize, to_pil_image
from os.path import join

from PIL import Image
from matplotlib import pyplot as plt

torch.manual_seed(0)

print('Using PyTorch version', torch.__version__)

# Preparing training and test sets

class_names = ['Normal', 'Viral Pneumonia', 'COVID']
root_dir = 'COVID-19 Radiography Database'
source_dirs = ['Normal', 'Viral Pneumonia', 'COVID']

if os.path.isdir(os.path.join(root_dir, source_dirs[1])):
    os.mkdir(os.path.join(root_dir, 'test'))

    for i, d in enumerate(source_dirs):
        os.rename(os.path.join(root_dir, d), os.path.join(root_dir, class_names[i]))

    for c in class_names:
        os.mkdir(os.path.join(root_dir, 'test', c))

    for c in class_names:
        images = [x for x in os.listdir(os.path.join(root_dir, c)) if x.lower().endswith('png')]
        selected_images = random.sample(images, 30) ##qui vado a scegliere la dimensione del testset
        for image in selected_images:
            source_path = os.path.join(root_dir, c, image)
            target_path = os.path.join(root_dir, 'test', c, image)
            shutil.move(source_path, target_path)

# Creating a custom dataset

class ChestXRayDataset(torch.utils.data.Dataset): 
  
    def __init__(self, image_dirs, transform): 
      
        def get_images(class_name):
            images = [x for x in os.listdir(image_dirs[class_name]) if x.lower().endswith('png')]
            print(f'Found {len(images)} {class_name} examples')
            return images
        
        self.images = {}
        self.class_names = ['Normal', 'Viral Pneumonia', 'COVID']
        
        for class_name in self.class_names:
            self.images[class_name] = get_images(class_name)
        # salvo, per tutte e tre le classi, le immagini corrispondenti nel dataset. 
            
        self.image_dirs = image_dirs
        self.transform = transform
        
    
    def __len__(self):
        return sum([len(self.images[class_name]) for class_name in self.class_names])
    
    
    def __getitem__(self, index): 
        # alla funzione di volta in volta. 
        class_name = random.choice(self.class_names)
        index = index % len(self.images[class_name]) 
        image_name = self.images[class_name][index] 
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path).convert('RGB')
        
        return self.transform(image), self.class_names.index(class_name)

# data augmentation  and image pre-processing

train_transform = torchvision.transforms.Compose([ #creiamo la lista di trasformazioni che ci interessa applicare alle immagini
    torchvision.transforms.Resize(size=(224, 224)), #resnet18 lavora con questi valori 
    torchvision.transforms.RandomHorizontalFlip(), #un po' di data augmentation 
    torchvision.transforms.ToTensor(), #importante perché pythorch lavora coi tensori
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #valori di riferimento per resnet18
])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# DataLoader

train_dirs = {
    'Normal': f'{root_dir}/Normal',
    'Viral Pneumonia': f'{root_dir}/Viral Pneumonia',
    'COVID': f'{root_dir}/COVID'
}

train_dataset = ChestXRayDataset(train_dirs, train_transform)
print (train_dataset)

test_dirs = {
    'Normal': f'{root_dir}/test/Normal',
    'Viral Pneumonia': f'{root_dir}/test/Viral Pneumonia',
    'COVID': f'{root_dir}/test/COVID'
}

test_dataset = ChestXRayDataset(test_dirs, test_transform)

batch_size = 6 
dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print('Number of training batches', len(dl_train))
print('Number of test batches', len(dl_test)) 

class_names = train_dataset.class_names


def show_images(images, labels, preds):
    plt.figure(figsize=(8, 4))
    for i, image in enumerate(images):
        plt.subplot(1, 6, i + 1, xticks=[], yticks=[])
        image = image.numpy().transpose((1, 2, 0)) 
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0., 1.)
        plt.imshow(image)
        col = 'green'
        if preds[i] != labels[i]:
            col = 'red'
            
        plt.xlabel(f'{class_names[int(labels[i].numpy())]}')
        plt.ylabel(f'{class_names[int(preds[i].numpy())]}', color=col)
    plt.tight_layout()
    plt.show()

images, labels = next(iter(dl_train))
show_images(images, labels, labels)


resnet18 = torchvision.models.resnet18(pretrained=True)
print(resnet18)

resnet18.fc = torch.nn.Linear(in_features=512, out_features=3)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet18.parameters(), lr=3e-5)

def show_preds():
    resnet18.eval() 
    images, labels = next(iter(dl_test)) 
    outputs = resnet18(images)
    #print(outputs)
    _, preds = torch.max(outputs, 1) 
    show_images(images, labels, preds)

# Training

def train(epochs):
    print('Starting training..')
    for e in range(0, epochs):
        print('='*20) 
        print(f'Starting epoch {e + 1}/{epochs}')
        print('='*20)

        train_loss = 0.
        val_loss = 0.

        resnet18.train() 

        for train_step, (images, labels) in enumerate(dl_train):
            optimizer.zero_grad() 
            outputs = resnet18(images)
            loss = loss_fn(outputs, labels) 
            loss.backward() 
            optimizer.step()
            train_loss += loss.item() 
            
            if train_step % 20 == 0: 
                print('Evaluating at step', train_step)

                accuracy = 0

                resnet18.eval() 

                for val_step, (images, labels) in enumerate(dl_test):
                    outputs = resnet18(images)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()
                    
                    _, preds = torch.max(outputs, 1) 
                    accuracy += sum((preds == labels).numpy()) 

                val_loss /= (val_step + 1)
                accuracy = accuracy/len(test_dataset)
                print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

                resnet18.train()

                if accuracy >= 0.95:
                    print('Performance condition satisfied, stopping..')
                    return


        train_loss /= (train_step + 1)

        print(f'Training Loss: {train_loss:.4f}')
    print('Training complete..')


%%time

train(epochs=30)
show_preds()

# Save the model

torch.save(resnet18.state_dict(), 'covid_classifier.pt')
model=torch.load ('covid_classifier.pt')

# Inference on a single image

resnet18 = torchvision.models.resnet18(pretrained=True)
resnet18.fc = torch.nn.Linear(in_features=512, out_features=3)

resnet18.load_state_dict(torch.load('covid_classifier.pt'))
resnet18.eval()
#model=resnet18()


def predict_image_class(image_path):
    image = Image.open(image_path).convert('RGB')
    image = test_transform(image)

    image = image.unsqueeze(0)
    output = resnet18(image)[0]
    probabilities = torch.nn.Softmax(dim=0)(output)
    probabilities = probabilities.cpu().detach().numpy()
    predicted_class_index = np.argmax(probabilities)
    predicted_class_name = class_names[predicted_class_index]
    return probabilities, predicted_class_index, predicted_class_name

