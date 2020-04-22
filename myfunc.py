import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import ImageFile, Image
from torchvision import datasets, transforms, models

def create_base_model(arch = 'vgg16'):
    ''' Create basic neural network model. Currently only supports vgg16 or AlexNet.
    Parameters:
        arch - base architecture type
    Returns:
        model - a pytorch model
    '''
    if arch == ('vgg16' or 'VGG'):
        model = models.vgg16(pretrained=True)
    elif arch == ('alexnet' or 'AlexNet'):
        model = models.alexnet(pretrained=True)
    else:
        print("Unrecognized architecture")
        sys.exit(0)
    return model

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def load_checkpoint(checkpoint):
    """
    Load trained network from checkpoint created using save_checkpoint(). Makes assumptions as to network architecture and optimizer.
    Parameters:
        checkpoint - path to checkpoint
    Returns:
        model - model set to eval mode.
    """
    checkpoint = torch.load(checkpoint)
    arch = checkpoint['arch']
    if (arch == 'VGG' or arch == 'vgg16'):
        model = models.vgg16(pretrained=True)
    elif checkpoint.arch == 'AlexNet':
        model = models.alexnet(pretrained=True)
    else:
        print("Checkpoint architecture not recognized. Abort.")

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    optimizer = optim.SGD(model.classifier.parameters(), lr = 0.001)
    optimizer = optimizer.load_state_dict(checkpoint['optim_dict'])
    model.load_state_dict(checkpoint['state_dict'])
    n_epochs = checkpoint["n_epochs"]
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def map_idx_to_class(class_to_idx, json_cat_labels = 'cat_to_name.json'):
    ''' Map class to category names and classes
    Parameters:
        class_to_idx - dict with class to index mapping
        json_cat_labels - json file with category to name mapping
    Returns:
        classes_map - mapping from index to class
    '''
    with open(json_cat_labels, 'r') as f:
        cat_to_name = json.load(f)

    n_classes = len(class_to_idx)
    classes_map = list("" for i in range(n_classes))
    for i in class_to_idx.values():         # map keys to list with 0 indexing
        classes_map[i] = list(class_to_idx.keys())[i]

    #names = classes.copy()
    #for idx in range(len(names)):
    #    names[idx] = cat_to_name[classes[idx]]

    #class_to_idx_dict = {}
    #class_to_idx_dict['class_to_idx'] = class_to_idx
    #class_to_idx_dict['classes'] = classes
    #class_to_idx_dict['names'] = names

    return classes_map, cat_to_name

def predict(image_path, model, classes_map, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    Parameters:
        image_path - path to image
        model - classifier model
        classes_map - mapping of index to class
        top_k - k top classifications, defaults to 5
    '''

    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path).unsqueeze(0)

    image = image.to(device)
    model = model.to(device)

    model.eval()
    #for param in model.parameters():
    #    param.requires_grad = False #param.requires_grad = False
    with torch.no_grad():
        log_preds = model(image)
        preds = torch.exp(nn.functional.log_softmax(log_preds, dim=1))
        probs, pred = torch.topk(preds, topk)
        #if use_cuda:  # maybe have to np.squeeze ?!
        probs = probs.cpu().numpy()[0]
        pred = pred.cpu().numpy()[0]
        #else:
        #    probs = probs.numpy()[0]
        #    pred = pred.numpy()[0]

    outcome = list("" for i in range(topk))
    for i in range(topk):
        outcome[i] = classes_map[pred[i]]
    return probs, outcome # predicted probs and class index

def process_data(train_dir, valid_dir, test_dir, batch_size = 20, num_workers = 0, data_transforms = None):
    ''' Preprocess data for loading
    Parameters:
        train_path - path to training data
        valid_path - path to training data
        test_path - path to training data
        batch_size - number of images per batch (default 30)
        num_workers - number of DataLoader threads (default 0)
    Returns:
        dataloaders - dict of DataLoaders for training, test, validation sets
        classes_map - mapping from index to class
    '''
    if data_transforms != None:
        data_transforms = data_transforms
    else:
        data_transforms = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(20),
                            transforms.RandomResizedCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225))])

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        "train_data": datasets.ImageFolder(train_dir, transform=data_transforms),
        "valid_data": datasets.ImageFolder(valid_dir, transform=data_transforms),
        "test_data" : datasets.ImageFolder(test_dir, transform=data_transforms)
        }

    # TODO: Using the image datasets and the tranforms, define the dataloaders
    dataloaders = {
        "train": torch.utils.data.DataLoader(image_datasets["train_data"], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "valid": torch.utils.data.DataLoader(image_datasets["valid_data"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test" : torch.utils.data.DataLoader(image_datasets["test_data"], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        }

    class_to_idx = image_datasets['train_data'].class_to_idx

    return dataloaders, class_to_idx

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    Args:
        img_path: path to an image

    Returns:
        image: image Tensor corresponding to the preprocessed image
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image)#.convert("RGB")

    size = 256 # Used for aspectratio resize
    input_transform = transforms.Compose([
                    transforms.Resize(size),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                        (0.229, 0.224, 0.225))])

    image = input_transform(image)#.unsqueeze(0)
    return image

def save_checkpoint(model, optimizer, class_to_idx, n_epochs = 1, n_classes = 102, checkpoint = 'checkpoint_cmd.pth'):#, args):
    """
    Saves trained network to a checkpoint called "classfier.pth".
    Parameters:
        model - Loader assumes vgg16
        optimizer - loader assumes vgg16
    Returns:
        none
    """
    model.cpu()
    if model.__class__.__name__ == "VGG":
        input_size = 25088
    elif  model.__class__.__name__ == "AlexNet":
        input_size = 9216
    torch.save({'arch': model.__class__.__name__,
                'n_epochs': n_epochs,
                'input_size': input_size,
                'output_size': n_classes,
                'class_to_idx': class_to_idx,
                'state_dict': model.state_dict(),
                'classifier': model.classifier,
                'optim_dict': optimizer.state_dict()
               },
                checkpoint)

def set_classifier(model, hidden_nodes, classifier=None):
    ''' Change/set model classifier.
    Parameters:
        model - neural network model
        classifier - defaults to in-function specified classifer (optional)
    Returns:
        model - a pytorch model
    '''
    if classifier == None:
        if model.__class__.__name__ == "VGG":
            in_features = 25088
        elif model.__class__.__name__ == "AlexNet":
            in_features = 9216
        else:
            print("Unrecognized architecture")
            sys.exit(0)
        classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_nodes, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=hidden_nodes, out_features=102, bias=True),
            nn.LogSoftmax(dim=1)
          )
    model.classifier = classifier
    return model

def set_device(use_gpu = True):
    ''' Select computational device based on user preference and availability.
    Parameters:
        use_gpu - request gpu usage
    Returns:
        device - torch device to use (cuda:0 or cpu)
    '''
    is_cuda = torch.cuda.is_available()
    if use_gpu and is_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return(device)

def train(model, optimizer, criterion, trainloader, validloader, n_classes, device, n_epochs = 1, do_validation = True):#, class_to_idx_map = None):
    ''' Train model.
    Parameters:
        model - neural network model
        optimizer - optimizer
        criterion - loss criterion
        trainloader - dataloader with training data set
        validloader - dataloader with validation data set
        n_classes - number of outputs
        device - computation device
        n_epochs - number of epochs
        do_validation = if validation should be performed
    Returns:
        model - trained neural network model
        optimizer - updated optimizer
    '''
    model.train()
    for epoch in range(1, n_epochs+1):

        # Track training loss
        train_loss, valid_loss = 0.0, 0.0

        ###################
        # Train the model #
        ###################
        # Model initially set to train
        for batch_i, (data,target) in enumerate(trainloader):
            # Move tensors to GPU when CUDA available
            data, target, model = data.to(device), target.to(device), model.to(device)
            # Clear gradients
            optimizer.zero_grad()
            # Forward pass: inputs to model -> prediction
            output = model(data)
            # Calculate batch loss
            loss = criterion(output,target)
            # Backward pass: Gradient of the loss with respect to model parameters
            loss.backward()
            # Perform a single optimization step/parameter update
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_i + 1)) * (loss.data - train_loss))

        if do_validation == True:
            class_correct = list(0. for i in range(n_classes))
            class_total = list(0. for i in range(n_classes))

            # Enter evaluation mode
            model.eval()
            # Iterate over test data
            for batch_i, (data, target) in enumerate(validloader):
                data, target = data.to(device), target.to(device)
                # Forward pass
                output = model(data)
                # Calculate batch loss
                loss = criterion(output, target)
                # Backward pass: Gradient of the loss with respect to model parameters
                loss.backward()
                valid_loss = valid_loss + ((1 / (batch_i + 1)) * (loss.data - valid_loss))
                # Update validation loss
                #valid_loss += loss.item()
                #val_count += len(target)

                # Convert output probabilities to predicted class
                _, pred = torch.max(output, 1)
                # Compare predictions to true label
                correct_tensor = pred.eq(target.data.view_as(pred))
                correct = np.squeeze(correct_tensor.cpu().numpy())
                # Calcluate test accuracy for each object class
                for i in range(len(target)):
                    label = target.data[i]
                    class_correct[label] += correct[i].item()
                    class_total[i] += 1

            print('\nValidation Accuracy: %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))

            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch,
                train_loss,
                valid_loss))
        else:
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                epoch,
                train_loss))

    return model, optimizer

def test(model, criterion, testloader, device, n_classes):
    ''' Test model and print outcome to the command line.
    Parameters:
        model - neural network model
        criterion - loss criterion
        test_loader - test set
        device - computation device
        n_classes - number of outputs
    '''
    # TODO: Do validation on the test set
    class_correct = list(0. for i in range(n_classes))
    class_total = list(0. for i in range(n_classes))

    # Enter evaluation mode
    model.eval()
    # Iterate over test data
    for data, target in testloader:
        data, target, model = data.to(device), target.to(device), model.to(device)
        # Forward pass
        output = model(data)
        # Calculate batch loss
        loss = criterion(output, target)
        # Convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # Compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        # Calcluate test accuracy for each object class
        for i in range(len(target)): #range(min(len(target),batch_size)): # target can be smaller than batch_size for the last iteration of dataloaders
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[i] += 1

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
