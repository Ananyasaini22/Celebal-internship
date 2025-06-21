def lowertriangle(rows):
    for i in range(rows):
        for j in range(i+1):
            print("*",end='')
        print()
def uppertriangle(rows):
    for i in range(rows):    
        for j in range(rows-i):
            print("*",end=" ")
        print()
def pyramid(rows):
    for i in range(rows):
        for j in range(rows-i-1):
            print(" ",end=" ")
        for k in range(2*i+1):
            print("*",end=" ")
        print()
rows=int(input("enter no. of rows- "))
print("lower triangle:")
lowertriangle(rows)
print("upper triangle:")
uppertriangle(rows)
print("pyramid: ")
pyramid(rows)