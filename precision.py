import csv
def pre():


# 读取第一个CSV文件
    data1 = []
    with open('csv/label/label01.csv', 'r') as file1:
        csv_reader = csv.reader(file1)
        for row in csv_reader:
        # 将科学计数法转换为整数
            row = list(map(lambda x: int(float(x)) if 'e' in x.lower() else x, row))
            data1.append(row)

# 读取第二个CSV文件
    data2 = []
    with open('outcome/csv/vall.csv', 'r') as file2:
        csv_reader = csv.reader(file2)
        for row in csv_reader:
        # 将科学计数法转换为整数
            row = list(map(lambda x: int(float(x)) if 'e' in x.lower() else x, row))
            data2.append(row)
    num =0
# 逐行对比数据
    data2=[int(x[0]) for x in data2]
    data1 =[int(x[0]) for x in data1]
    for i in range(len(data2)):
        row1 = data1[i]
        row2 = data2[i]
        
        if row1 == row2:
            num+=1  
    print("precision:")
    print(num/len(data1))
