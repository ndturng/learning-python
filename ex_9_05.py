f_name = input('Enter a file name: ')

try:
    f_open = open(f_name)
except:
    print('Can not open', f_name)
    exit()


lst = dict()

for line in f_open:
    
    if line.startswith('Author:'):   
        line = line.rstrip()
        words = line.split()
        
        a = words[1].find('@')
        domain = words[1][a+1:]
        
        
        lst[domain] = lst.get(domain,0) + 1

        
print(lst)
    
        


