import torch
from iresnet import iresnet50

batch_size = 10000
channels = 3
height = 112
width = 112
inputs = torch.randn(batch_size, channels, height, width)



model = iresnet50()
model.to('cuda')
inputs = inputs.to('cuda')
pretrained_path = '/home/data/ckpt/backbone_iresnet50.pth'
if pretrained_path:
    state_dict = torch.load(pretrained_path)
    model.load_state_dict(state_dict)

model.eval()


with torch.no_grad():
    outputs = model(inputs)

