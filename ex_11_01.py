import re
f_name = open('mbox.txt')

re_name = input('Enter a regular expression: ')

c = 0

for line in f_name:
    line = line.rstrip()
    if re.search(re_name, line):
        c += 1

print('mbox.txt had', c ,'lines that matched', re_name)