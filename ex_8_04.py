list_out = []

fhand = open('romeo.txt')
for line in fhand:
    w = line.split()
    list_out.append(w[0])
    for i in (range(len(w))):
        if w[i] in list_out:
            continue
        else: 
            list_out.append(w[i])

list_out.sort()   
    
print(list_out)


#     for i in w:
#         if len(list_out) == 0:
#             list_out.append(i)
#         if i in list_out:
#             continue
#         elif i < list_out[0]:
#             list_out[0].append(i)
        

# print(list_out)            