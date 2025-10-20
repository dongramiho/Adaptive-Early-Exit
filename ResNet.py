from torchvision.models import resnet101
from torchinfo import summary
model = resnet101(pretrained=False)
summary(model, input_size=(1,3,224,224))