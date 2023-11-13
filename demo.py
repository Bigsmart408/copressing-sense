import csv

# 读取CSV文件
with open('csv/data/vall.csv', 'r') as file:
    reader = csv.reader(file)
    rows = list(reader)

# 获取TICKS列的索引
header = rows[0]
ticks_index = header.index('TICKS')

# 移除TICKS列
header.pop(ticks_index)
for row in rows:
    row.pop(ticks_index)

# 写入处理后的数据到新的CSV文件
with open('csv/data/all.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(rows[1:])