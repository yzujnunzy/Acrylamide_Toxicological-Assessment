import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
from models.densenet import densenet121
import matplotlib.pyplot as plt
from models.resnet import resnet34, resnet50
device = torch.device('cuda:0')
import seaborn as sns
import argparse
def main(args):
    model = resnet34(num_class=8).to(device)
    weights_path = args.weights_path
    model.load_state_dict(torch.load(weights_path,map_location=device,weights_only=True))
    data_transform = transforms.Compose(
                [transforms.Resize(256),
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def evaluate_model(model, validate_loader):
        all_predictions = []
        all_labels = []
        model.eval()  # Setting up the model for evaluation mode
        predictions = []
        labels = []
        with torch.no_grad():  # No need to calculate the gradient
            for data_test in validate_loader:
                test_images, test_labels = data_test
                outputs = model(test_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]  # dim = 1 means that the output is the maximum value of the row, if dim = 0 means that the output is the maximum value of the column.
                all_predictions.extend(predict_y.cpu().numpy())
                all_labels.extend(test_labels.cpu().numpy())
                # Calculate the confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions, labels=range(8))
        return conf_matrix
    path =args.data
    val_data = datasets.ImageFolder(root=path,transform = data_transform)
    val_loader = torch.utils.data.DataLoader(val_data,batch_size = 32,shuffle=False)
    conf_matrix = evaluate_model(model,val_loader)

    class_names = ['0','100','1000','200','300','400','50','500']
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',type=str,default='XXX')
    parser.add_argument('--weights_path',type=str,default='XXX')

    opt = parser.parse_args()
    main(opt)