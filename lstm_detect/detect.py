import torch
from torch.utils.data import DataLoader

from modeling import Model
from dataset_torch import DetectDataSet
from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    model = Model(input_size=1, hidden_dim=256, num_layers=2, num_classes=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load("model.pkl"))
    model.to(device)
    model.eval()

    dataset = DetectDataSet("C:\Program Files\code\Python\project788067-124730-pre\cpu.csv")
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False)

    export_flag = True  # 假设导出标志为真
    loss_func = torch.nn.L1Loss()
    len =0
    begin = 0
    end =0
    temp =0
    a=0
    for batch in dataloader:
        input, label, timestamp = batch
        input = input.to(device)
        label = label.to(device)
        # 调整输入维度为二维
        input = input.unsqueeze(-1)

        if export_flag:
            # 导出模型为 ONNX
            torch.onnx.export(model, input, "model.onnx", input_names=["input"], output_names=["output"])

            export_flag = False


        output = model(input)
        loss = loss_func(output, label)
        len+=1
        temp =timestamp[0]

        if loss.item() > 480:
            time = datetime.fromtimestamp(int(float(timestamp[0])+end-begin))
            print(f"{time} abnomal CPU usage, {loss}")
            a+=1
    print(len)
    print(a)



