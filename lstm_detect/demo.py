from sched import scheduler
import toml
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from modeling import Model
from dataset import DetectDataSet

TIME_STEPS = 80
BATCH_SIZE = 64
NUM_EPOCHS = 500

if __name__ == "__main__":
    with open('input_data.toml') as f:  # load data path
        config = toml.loads(f.read())

    normal_data_path = config['normal']['data']
    train_data = DetectDataSet(normal_data_path[0])
    print("Data initialization done.")

