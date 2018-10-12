#coding:utf-8
import shutil
readDir = "C:/CCF/TF-IDF+Kmeans/data/微贷网.txt"
writeDir = "C:/CCF/TF-IDF+Kmeans/data/微贷网去重.txt"
#txtDir = "/home/fuxueping/Desktop/１"
lines_seen = set()
outfile=open(writeDir,"w")
f = open(readDir,"r")
for line in f:
    if line not in lines_seen:
        outfile.write(line)
        lines_seen.add(line)
outfile.close()
print('success')
