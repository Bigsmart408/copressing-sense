import torch
import csv
import datetime
import numpy as np

TIME_STEPS = 80


def create_sequences(sequence, time_steps=TIME_STEPS):
    X = []
    for i in range(len(sequence) - time_steps + 1):
        X.append(sequence[i:i + time_steps])
    return X

class DataSet(object):
    def __init__(self, data_path):
        self.data_path = data_path
        timestamp = []
        cpu = []
        mem =[]
        netsent=[]
        netrev = []
        with open(data_path) as f:
            csv_reader = csv.reader(f)
            for id, row in enumerate(csv_reader):
                if id == 0:
                    continue
                if (len(row) == 0):
                    print(id)
                    break
                a = float(row[0])

                dt = datetime.datetime.fromtimestamp(a)

                formatted_date = dt.strftime('%Y-%m-%d %H:%M:%S')
                element = datetime.datetime.strptime(
                    formatted_date, "%Y-%m-%d %H:%M:%S")
                t = datetime.datetime.timestamp(element)
                timestamp.append(t)
                cpu.append(row[1])
                mem.append(row[2])
                netsent.append(row[3])
                netrev.append(row[4])

        self.data = {'timestamp': timestamp, 'cpu': cpu,'mem':mem,'net-sent':netsent,'net-rev':netrev}
        self.data = create_sequences(self.data['cpu','mem','net-set','net-rev'])
        input1 = torch.Tensor(self.data[0]).unsqueeze(-1)
        label1 = torch.Tensor(self.data[1]).unsqueeze(-1)
        input2 = torch.Tensor(self.data[2]).unsqueeze(-1)
        label2 = torch.Tensor(self.data[3]).unsqueeze(-1)
        input3 = torch.Tensor(self.data[4]).unsqueeze(-1)
        label3 = torch.Tensor(self.data[5]).unsqueeze(-1)
        input4 = torch.Tensor(self.data[6]).unsqueeze(-1)
        label4 = torch.Tensor(self.data[7]).unsqueeze(-1)
        self.data = {'input-cpu': input1, 'label-cpu': label1,'input-mem':input2,'label-mem':label2,'input-net-sent':input3,'label-net-sent':label3,'input-net-rev':input4,'label-net-rev':label4}
        self.length = len(self.data['input-cpu'])

    def __getitem__(self, index):
        return self.data['input1'][index], self.data['label1'][index],self.data['input2'][index], self.data['label2'][index],self.data['input3'][index], self.data['label3'][index],self.data['input4'][index], self.data['label4'][index]

    def __len__(self):
        return self.length


class DetectDataSet(object):
    def __init__(self, data_path):
        print("init")
        self.data_path = data_path
        with open(data_path) as f:
            csv_reader = csv.reader(f)
            timestamp = []
            value = []

            for id, row in enumerate(csv_reader):
                if id == 0:
                    continue
                if (len(row) == 0):
                    print(id)
                    break
                timestamp.append(row[0])
                value.append(row[1])

        self.data = {'timestamp': timestamp, 'value': value}

        tmp_data = create_sequences(self.data['value'])
        input, label = tmp_data
        # input = np.expand_dims(tmp_data[0], axis=0)
        # label = np.expand_dims(tmp_data[1], axis=0)
        self.data = {'input': input, 'label': label,
                     'timestamp': timestamp[79:-1]}
        self.length = len(self.data['input'])

    def __getitem__(self, index):
        return self.data['input'][index], self.data['label'][index], self.data['timestamp'][index]

    def __len__(self):
        return self.length
