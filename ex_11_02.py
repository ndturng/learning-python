import re

f_name = input('Enter a file name: ')
# if len(f_name) == 0:
#     f_name = 'mbox-short.txt'

try:
    f_open = open(f_name)
except:
    print('Can not open', f_name)
    exit()

c = 0
t = 0
for line in f_open:
    line = line.rstrip()
    x = re.findall('New Revision: ([0-9]+)', line)
    if len(x) > 0:
        c += 1
        for i in x:
            t += float(i)

print(t/c)

   