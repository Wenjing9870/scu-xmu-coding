# -*- coding:utf-8 -*-
import io


file = io.open('comment.txt', 'r', encoding='utf-8')
line = file.readlines()
line_new = []
line_new1 = []
for idx, line0 in enumerate(line):
    # print(line0[0])
    line_new = line0.strip().split('\t')
    for i in line_new:
        print i
        i = i.encode('utf-8')
        line_new1.append(i)
    # line_new.append(line0)
    print line_new1
    if idx == 1:
        break
