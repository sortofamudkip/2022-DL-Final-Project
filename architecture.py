import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch, torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms


class JakeDemoNet(nn.Module):
    def __init__(self):
        super(JakeDemoNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(7056, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        # softmax at the end
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


def resnet_pretrained():
    m = models.resnet18(pretrained=True)
    num_ftrs = m.fc.in_features
    m.fc = nn.Linear(num_ftrs, 2)
    return m


def vgg16_pretrained():
    # TODO: Need to transform the data
    model_vgg16 = models.vgg16(pretrained=True)  # choosing the model
    num_features = model_vgg16.classifier[6].in_features
    features = list(model_vgg16.classifier.children())[:-1]  # Remove last layer
    features.extend([nn.Linear(num_features, 2)])  # Add our layer with 4 outputs
    model_vgg16.classifier = nn.Sequential(*features)
    return model_vgg16


# A model consists of image transformation suitable to the model + the actual model itself
# Returning the type and not the instance makes parameterization a bit easier.
# From the outside you can then call: models['resnet_pretrained']()
models_dict = {
    "jake_demo": ([transforms.ToTensor()], JakeDemoNet),
    "resnet_pretrained": (
        [
            transforms.Resize(224),
            *(
                [
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(brightness=0.5),
                            transforms.ColorJitter(contrast=0.5),
                            transforms.ColorJitter(saturation=0.5),
                            transforms.ColorJitter(hue=0.5),
                        ],
                        0.5,
                    )
                ]
            ),
            *(
                [
                    transforms.RandomApply(
                        [
                            transforms.RandomRotation((0, 0)),
                            transforms.RandomHorizontalFlip(p=1),
                            transforms.RandomVerticalFlip(p=1),
                            transforms.RandomRotation((90, 90)),
                            transforms.RandomRotation((180, 180)),
                            transforms.RandomRotation((270, 270)),
                        ],
                        0.5,
                    )
                ]
            ),
            transforms.ToTensor(),
        ],
        resnet_pretrained,
    ),
    "vgg16_pretrained": (
        [transforms.Resize(224), transforms.ToTensor()],
        vgg16_pretrained,
    ),
}
