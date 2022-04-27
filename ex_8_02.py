fhand = open('mbox-short-bug.txt')

for line in fhand:
    words = line.split()
    if len(words) < 3 : continue  # if len(words) == 0 : continue -> will get traceback 
    if words[0] != 'From' : continue
   
    print(words[2]) 

