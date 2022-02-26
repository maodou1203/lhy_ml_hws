import torch
import torch.nn as nn

# class classifier(nn.Module):
#     def __init__(self):
#         super(classifier, self).__init__()
#         # The arguments for commonly used modules:
#         # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
#         # torch.nn.MaxPool2d(kernel_size, stride, padding)
#
#         # input image size: [3, 128, 128]
#
#         self.cnn_layer = nn.Sequential(
#             nn.Conv2d(3, 64, 3, 1, 1), #[64, 128, 128]
#             nn.BatchNorm2d(64), #[64, 128, 128]
#             nn.ReLU(), #[64, 128, 128]
#             nn.MaxPool2d(2, 2, 0), #[64, 64, 64]
#
#             nn.Conv2d(64, 128, 3, 1, 1), #[128, 64, 64]
#             nn.BatchNorm2d(128), #[128, 64, 64]
#             nn.ReLU(), #[128, 64, 64]
#             nn.MaxPool2d(2, 2, 0), #[128, 32, 32]
#
#             nn.Conv2d(128, 256, 3, 1, 1), #[256, 32, 32]
#             nn.BatchNorm2d(256), #[256, 32, 32]
#             nn.ReLU(), #[256, 32, 32]
#             nn.MaxPool2d(4, 4, 0), #[256, 8, 8]
#
#         )
#
#         self.fc_layers = nn.Sequential(
#             nn.Linear(256 * 8 * 8, 256),
#             nn.ReLU(),
#             nn.Linear(256,256),
#             nn.ReLU(),
#             nn.Linear(256,11),
#
#         )
#
#     def forward(self, x):
#         # input (x): [batch_size, 3, 128, 128]
#         # output: [batch_size, 11]
#
#         # Extract features by convolutional layers.
#         x = self.cnn_layer(x)
#         # The extracted feature map must be flatten before going to fully-connected layers.
#         x = x.flatten(1)
#
#         # The features are transformed by fully-connected layers to obtain the final logits.
#         x = self.fc_layers(x)
#         return x
#
#
class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x = self.cnn_layers(x)

        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x
