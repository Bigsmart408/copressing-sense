from sched import scheduler
import toml
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, dataloader, ConcatDataset
from torch.optim.lr_scheduler import ExponentialLR

from modeling import Model
from dataset_torch import DataSet

TIME_STEPS = 80
BATCH_SIZE = 64
NUM_EPOCHS = 10


device = 'cuda'

if __name__ == "__main__":
    with open('input_data.toml') as f:  # load data path
        config = toml.loads(f.read())

    normal_data_path = config['normal']['data']
    normal_data = DataSet(normal_data_path[0],0)
    abnormal_data = DataSet(normal_data_path[1],1)
    dataset = ConcatDataset([normal_data, abnormal_data])
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True)
    print("Data initialization done.")

    model = Model(input_size=1, hidden_dim=256, num_layers=2, num_classes=1)

    loss_func = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.96)

    # Training loop
    for epoch_id in range(NUM_EPOCHS):
        cur_batch_loss = 0

        for batch in dataloader:
            input, label, target = batch  # Modified to include target
            input = input.to(device)
            label = label.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(input)
            loss = loss_func(output, target)  # Modified to use target
            cur_batch_loss += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()

        cur_batch_loss /= len(dataloader)

        print(f"Epoch {epoch_id + 1} Loss: {cur_batch_loss}, cur_lr: {scheduler.get_last_lr()}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = []
        labels = []
        for batch in dataloader:
            input, label, target = batch  # Modified to include target
            input = input.to(device)
            output = model(input)
            preds.extend(torch.argmax(output, dim=1).cpu().tolist())
            labels.extend(label.tolist())

    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)
    print(f"Accuracy: {accuracy}")