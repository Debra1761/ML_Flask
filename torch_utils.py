import torch
import io
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image #convert image too tensor

#load model

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out

input_size = 784 # 28x28
hidden_size = 500 
num_classes = 10
num_epochs = 2 

model = NeuralNet(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load("mnist_ffn.pth"))
model.eval()

#image --> tensor
def transform_image(image_bytes):
    transform = transforms.Compose(
                                    [ transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((28,28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,),(0.3081,)) ]
                                   )
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

#predict


def get_predictions(image_tensor):
        images = image_tensor.reshape(-1, 28*28)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        return predicted
