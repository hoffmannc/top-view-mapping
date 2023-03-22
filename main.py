import torch

from src.data.dataset import NuScencesMaps
from src.model.network import PyrOccTranDetr_S_0904_old_2DPosEncBEV as Model


path = "/Users/colinhoffmann/Desktop/nuscenes"
split = "train_split"
dataset = NuScencesMaps(path, split)

model = Model(
    num_classes=4,
    frontend="resnet50",
    grid_res=0.5,
    pretrained=True,
    img_dims=[1600, 900],
    z_range=[1.0, 6.0, 13.0, 26.0, 51.0],
    h_cropped=[60.0, 60.0, 60.0, 60.0],
    dla_norm="GroupNorm",
    additions_BEVT_linear=False,
    additions_BEVT_conv=False,
    dla_l1_n_channels=32,
    n_enc_layers=2,
    n_dec_layers=2,
)

image, calib, label, mask = dataset[1]
grid = dataset.grid

image = image.unsqueeze(0)
calib = calib.unsqueeze(0)
grid = grid.unsqueeze(0)

out = model(image, calib, grid)

device = torch.device("mps")
model.to(device)

test = 0
