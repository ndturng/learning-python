f_name = input('Enter a file name: ')

try:
    f_open = open(f_name)
except:
    print('Can not open', f_name)
    exit()


lst = dict()

for line in f_open:
    
    if line.startswith('From:'):
        continue
    elif line.startswith('From'):   
        line = line.rstrip()
        words = line.split()
        try:
            lst[words[1]] = lst.get(words[1],0) + 1
        except: continue

a = 0
b = 0

for key, value in lst.items():   	# another way:
    if value > a:			# inverse = [(value, key) for key, value in lst.items()]
        a = value			# print(max(inverse)[1], max(inverse)[0])
        b = key

print(b, a)        





