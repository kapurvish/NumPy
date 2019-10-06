# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:38:16 2019

@author: Vishal Kapur
"""
from scipy import ndimage
from scipy import misc
import numpy as nm
import sys
import matplotlib.pyplot as plt
array_1 = nm.array([1,2,3,4])
print(array_1)
#[1 2 3 4]

number=[9,6,7,7]
array_2 = nm.array(number)
print(array_2)
#[9 6 7 7]

arr_zeros = nm.zeros((3,4))
print(arr_zeros)
#An elements of array equal to zeros with 3 rows and 4 columns
#[[0. 0. 0. 0.]
# [0. 0. 0. 0.]
# [0. 0. 0. 0.]]


arr_ones = nm.ones((3,4),dtype=nm.int16)
print(arr_ones)
#An elements of array equal to ones with 3 rows and 4 columns. We can specify the 
#data type as well, here its int16
#[[1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]]

arr_empty = nm.empty((2,3))
print(arr_empty)
#To create an empty array.
#[[0. 0. 0.]
# [0. 0. 0.]]

arr_diag = nm.eye(3,3)
print(arr_diag)
#This allows us to create 1 along the diagonal elements.
#[[1. 0. 0.]
# [0. 1. 0.]
# [0. 0. 1.]] 


arr_range = nm.arange(2,28,2)
print(arr_range)
#Here the end value is exclusive, we don't have 28 in the output.Step size is 2 starting 
#from number 2
#[ 2  4  6  8 10 12 14 16 18 20 22 24 26]

arr_2D = nm.array([(3,4,5),(7,8,9)])
print(arr_2D)
#2 Dimentional array, to pass list of list, with two rows and three columns.
#[[3 4 5]
# [7 8 9]]
print(arr_2D.shape)
#(2, 3)  2 rows and three columns

abc = nm.arange(8)
print(abc)
#[0 1 2 3 4 5 6 7]
#Here it creates the array starting form 0 till 6, 7 is not included.
#we have also reshaped the array, to have 2 rows and 4 coulmns.
abc_arr = abc.reshape(2,4)
print(abc_arr)
#[[0 1 2 3]
# [4 5 6 7]]

#To create a new array with exactly the same dimensions but pre fill the array with ones.
abc_arr_same = nm.ones_like(abc_arr)
print(abc_arr_same)
#[[1 1 1 1]
# [1 1 1 1]]

array_3d = nm.arange(24).reshape(2,3,4)
print(array_3d)
print(array_3d.shape)
#Here its a three dimentional array, 2 array of size 3*4.
#(2, 3, 4)
#[[[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
#
# [[12 13 14 15]
#  [16 17 18 19]
#  [20 21 22 23]]]


nm.set_printoptions(threshold=sys.maxsize)
print(nm.arange(400).reshape(20,20))
#This will print the complete array, without any ellipsis ...
#[[  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
#   18  19]
# [ 20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37
#   38  39]


a = nm.array([22,40,25])
b = nm.array([5,5,5])
c = a+b
print(c)
#[27 45 30]

d = a-b
print(d)
#[17 35 20]

e = a*b
print(e)
#[110 200 125]


result = a<35
#Here a<35 for first and last element, so its 'True'
print(result)
#[ True False  True]

aa = nm.array([[1,1],[0,1]])
bb = nm.array([[2,0],[3,4]])
cc= aa * bb
print(cc)
#Here is mutiplying each element with the other element.
#[[2 0]
# [0 4]]

print(aa.dot(bb))
print(nm.dot(aa,bb))
#we are doing the matrix multiplication, sum(element at a particular row * element in column).
#[[5 4]
# [3 4]]

ages = nm.array([3,6,8,9])
print(ages.sum())
#26
print(ages.min())
#3
print(ages.max())
#9


zz = nm.arange(12).reshape(3,4)
print(zz)
#[[ 0  1  2  3]
# [ 4  5  6  7]
# [ 8  9 10 11]]
print(zz.sum(axis=0))
#Here axis=0, sum of all elements along all the columns
#Here axis=1, sum of all elements along all the rows
#[12 15 18 21]

angles = nm.array([0,30,45,60,90])
angles_radians = nm.radians(angles)
#Convert angle in radians
print(nm.sin(angles_radians))
#[0.         0.5        0.70710678 0.8660254  1.        ]

#We can also do the statistical functions on the array.

score = nm.array([0,30,45,60,90])
print(nm.mean(score))
#45
#To find the mean of the score and also median

salary = nm.genfromtxt('salary.csv',delimiter = ',')
#Get slalry from the csv file in the array.
print(salary)
print(salary.shape)
#(1147,)

abc = nm.arange(11) ** 2
print(abc)
#[  0   1   4   9  16  25  36  49  64  81 100]

print(abc[2])
#4

#Last element start from -1
print(abc[-2])
#81

print(abc[2:7])
#Here last '7' postion is not included
#[ 4  9 16 25 36]

print(abc[2:])
#Go till the end of the array
#[  4   9  16  25  36  49  64  81 100]


print(abc[:7])
#Start from begning and go till index 6, 7 is not included
#[ 0  1  4  9 16 25 36]

print(abc[:11:2])
#Start with beginning and go till 10th poston and step size 2
#[  0   4  16  36  64 100]

print(abc[::-1])
#[100  81  64  49  36  25  16   9   4   1   0]
#We need all elemnt, but start from end, -1 element

student = nm.array([['vishal','parma','rohan'], 
                    ['34','23','44'],
                    ['11','22','33']] 
                   )
print(student.shape)
print(student[0])
print(student[1])
print(student[2])
#['vishal' 'parma' 'rohan']
#['34' '23' '44']
#['11' '22' '33']

print(student[0,2])
#rohan

print(student[0:2,2:4])

#[['rohan']
# ['44']]

print(student[:,1:2])
#[['parma']
# ['23']
# ['22']]

print(student[-1,:])
#['11' '22' '33']

print(student[-3:-1,-2:])
#['11' '22' '33']
#[['parma' 'rohan']
# ['23' '44']]

#Iterate elements in a array
xyz = nm.arange(11) ** 2
print(xyz)
for i in xyz:
    print(i)
#0
#1
#4
#9
#16
#25
#36
#49
#64
#81
#100
    

student1 = nm.array([['vishal','parma','rohan'], 
                    ['34','23','44'],
                    ['11','22','33']] 
                   )    
for i in student1:
    print(i)
#['vishal' 'parma' 'rohan']
#['34' '23' '44']
#['11' '22' '33']

#To access each element in a 2D array, we use flatten, by default it row wise
for element in student1.flatten():
    print(element)
#vishal
#parma
#rohan
#34
#23
#44
#11
#22
#33

for element in student1.flatten(order='F'):
    print(element)
#This gives the column wise flatten
#vishal
#34
#11
#parma
#23
#22
#rohan
#44
#33
    
vk = nm.arange(12).reshape(3,4)
print(vk)
#[[ 0  1  2  3]
# [ 4  5  6  7]
# [ 8  9 10 11]]
        
#This iterates over the row. use order='F' to iterate over the columns
for i in nm.nditer(vk):
    print(i)
#0
#1
#2
#3
#4
#5
#6
#7
#8
#9
#10
#11
    
#Difference between flatten and nditer: flatten returns a result array and nditer allows to iterate over
#the elements   
    
for i in nm.nditer(vk,order = 'F',flags =['external_loop']):
    print(i)
#   i[...]= i * i
#Here every column is expressed as a 1D array.
#ValueError: assignment destination is read-only
#We can use the parameter: op_flags=['readwrite'], so that we can write as well using nditer
#It is a read only nditer, we can not assign anything i it
#[0 4 8]
#[1 5 9]
#[ 2  6 10]
#[ 3  7 11]   

country = nm.array([('nepal','india','bhutan'),
                   ('russia','usa','africa'),
                   ('paris','canada','dubai')
                   ])
print(country)
print(country.shape)
#(3, 3)

print(country.ravel())
#It flattens the array, this flatten the array to 1D vector, a copy of the array is made using ravel and 
#copy is made only if needed, if the shape has changed.
#['nepal' 'india' 'bhutan' 'russia' 'usa' 'africa' 'paris' 'canada' 'dubai']

print(country.T)
#This gives the transpose of a array. ravel can also be used as a.T.ravel
#[['nepal' 'russia' 'paris']
# ['india' 'usa' 'canada']
# ['bhutan' 'africa' 'dubai']]

country = nm.array([('nepal','india','bhutan','japan'),
                   ('russia','usa','africa','romania'),
                   ('paris','canada','dubai','yemen')
                   ])
print(country.reshape(2,6))
#[['nepal' 'india' 'bhutan' 'japan' 'russia' 'usa']
# ['africa' 'romania' 'paris' 'canada' 'dubai' 'yemen']]


#Use of -1 in the reshape parameters.
countries = nm.array(['germany','india','pakistan','nepal','bhutan','indonesia'])
#Here numpy will create 3 columns and the rows depends on the number of elements in it.
print(countries.reshape(-1,3))
#[['germany' 'india' 'pakistan']  
# ['nepal' 'bhutan' 'indonesia']]

print(countries.reshape(3,-1))
#Here numpy will create 3 rows and the columns depends on the number of elements in it.
#[['germany' 'india']
# ['pakistan' 'nepal']
# ['bhutan' 'indonesia']]

test_array = nm.arange(10)
print(nm.split(test_array,2))
#here we have split into 2 equal size array
#[array([0, 1, 2, 3, 4]), array([5, 6, 7, 8, 9])]
print(nm.split(test_array,[2,5]))
#here we are saying split the array at 2nd and 5th postion.
#[array([0, 1]), array([2, 3, 4]), array([5, 6, 7, 8, 9])]

country_group = nm.array([('nepal','india','bhutan','japan'),
                   ('russia','usa','africa','romania'),
                   ('paris','canada','dubai','yemen')
                   ])   
p1,p2 = nm.hsplit(country_group,2)
print(p1)
#This horizontally split the array into two parts. p1 contains one part and p2 contains the other part
#Same we have vsplit to vertical split the array.
#[['nepal' 'india']
# ['russia' 'usa']
# ['paris' 'canada']]

f = misc.face()
print(f.shape)
#(768, 1024, 3)
print(type(f))
#<class 'numpy.ndarray'>

#print(f)
#[ 32  55  13]
#  [ 26  49   3]
#  [ 43  70  19]

#plt.imshow(f)
#To see the image.

a = f[200:,700:,:]
#plt.imshow(a)
#This will select the specific part of the image.



#view is a shallow copy of the array.Any edits made to the copy will also reflect to the original array.
#
fruit = nm.array(["apple","mango","kiwi","banana",])
print(fruit)
#['apple' 'mango' 'kiwi' 'banana']

basket_1 = fruit.view()
basket_2 = fruit.view()
print(basket_1)
print(basket_2)
#['apple' 'mango' 'kiwi' 'banana']
#['apple' 'mango' 'kiwi' 'banana']

print(id(fruit))
print(id(basket_1))
print(id(basket_2))

#2264168417440
#2264192463376
#2263995791680


basket_2[0] = "chiko"
print(basket_2)
#['chiko' 'mango' 'kiwi' 'banana']
print(fruit)
#['chiko' 'mango' 'kiwi' 'banana']
#Here as we changed the basket_2, the value in basket_1 and fruit array has also beenc changed.
print(basket_1)
#['chiko' 'mango' 'kiwi' 'banana']



fruits = nm.array(["apple","mango","kiwi","banana",])
print(fruits)
#['apple' 'mango' 'kiwi' 'banana']

basket1 = fruits.copy()
#This is nota view or the shallow copy.
#['apple' 'mango' 'kiwi' 'banana']
print(basket1)

basket1[0] = "orange"
print(basket1)
print(fruits)
#['orange' 'mango' 'kiwi' 'banana']
#['apple' 'mango' 'kiwi' 'banana']

arr_1 = nm.arange(12)**2
print(arr_1)
#[  0   1   4   9  16  25  36  49  64  81 100 121]
print(arr_1[2],arr_1[5],arr_1[7])
#4 25 49

index_arr = [2,5,7]

print(arr_1[index_arr])
#[ 4 25 49]

arr_2 = nm.array([[2,3],[7,8]])
print(arr_2)
#[[2 3]
# [7 8]]

print(arr_1[arr_2])
#Here we are taking the arr_1 and the index we are using as arr_2 and is valid.
#[[ 4  9]
# [49 64]]

new_food = nm.array([["noodle","chinese","clove","indian"],
                     ["south","rice","wheat","bread"],
                     ["cake","pastry","juice","chocklate"]])

row = nm.array([[0,0],[2,2]])
col = nm.array([[0,3],[0,3]])

print(new_food[row,col])
#[['noodle' 'indian']
# ['cake' 'chocklate']]

new_food[row,col] = "00000"
print(new_food)
#With this assignment operator, value "0000' at the four corners
#[['00000' 'chinese' 'clove' '00000']
# ['south' 'rice' 'wheat' 'bread']
# ['00000' 'pastry' 'juice' '00000']]

ab_arr = nm.array([3,6,8,23,56,77])
index_1 = [2,3,5]
ab_arr[index_1] = 99
print(ab_arr)
#[ 3  6 99 99 56 99]

ac= nm.tile(2,(4,5))
#It takes 2 and creates an array of dimentions 4,5 and fill 2 at all postions.
print(ac)
#[[2 2 2 2 2]
# [2 2 2 2 2]
# [2 2 2 2 2]
# [2 2 2 2 2]]

ax= nm.tile(nm.arange(10),(4,2))
print(ax)
#[[0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9]
# [0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9]
# [0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9]
# [0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9]]

print(nm.repeat([1,2],5))
#It takes the list, 1 and 2 and repeat 5 times in array.
#[1 1 1 1 1 2 2 2 2 2]

#Structured array function
struc_arr = nm.array([('abs','.8'),('nonabs','.7'),('hero','.9')],dtype=[('label','a8'),('prob','f8')])
print(struc_arr['label'])
print(struc_arr['prob'])
#[b'abs' b'nonabs' b'hero']
#[0.8 0.7 0.9]

#Broadcasting of the array, we do in another language we need to iterate with loop and then count.
#It broadcast the operations along all the list.
aa = nm.array([2,3,5,6])
bb = nm.array([1,5,9,8])
print(aa +bb)
#[ 3  8 14 14]

#Broadcasting rules: check each dimensions of the numpy array, same dimentions of both array or one of them is 1
xx =nm.array(nm.arange(100).reshape(20,5))
yy =nm.array(nm.arange(20).reshape(20,1))
print(xx+yy)
#It will work as we have both have rows as '20' and in the second array, 
#we have 1 as columns, we have 5 and 1, it i will work .

ss = nm.ones((9,8,1,3,3))
sd = nm.ones((1,8,9,1,3))
print(ss + sd)
#yes it works as either the dimensions are same ot either has 1 as dimentions
#(9,1) or (8,8) or (1,9) or (3,1) or (3,3)

new_arr  = nm.arange(10)
print(new_arr)
print(new_arr.shape)
#[0 1 2 3 4 5 6 7 8 9]
#(10,)
#Its a one dimention array, newaxis will add a new dimention to the array
print(new_arr[:,nm.newaxis])
print(new_arr[:,nm.newaxis].shape)
#It creates an array and create a artificial second dimensions.
#this is used in keras to add an additonal dimensions for calculations.
#[[0]
# [1]
# [2]
# [3]
# [4]
# [5]
# [6]
# [7]
# [8]
# [9]]
#(10, 1)

sd = nm.arange(15).reshape(3,5,1)
print(sd)
#[[[ 0]
#  [ 1]
#  [ 2]
#  [ 3]
#  [ 4]]
#
# [[ 5]
#  [ 6]
#  [ 7]
#  [ 8]
#  [ 9]]
#
# [[10]
#  [11]
#  [12]
#  [13]
#  [14]]]
axr =sd[1,...]
print(axr)
print(axr.shape)
#[[5]
# [6]
# [7]
# [8]
# [9]]
#(5,1)
#'...' is a ellipsis in numpy
#take first row and and everything afterwards
#print(sd[1,...]) and print(sd[1,:,:]) works the same

#there are scenarios where we get data, in which the last dimention is meaningless, we can use the
#squeeze to remove all dimentions that are one removed.
arr_sq = axr.squeeze()
print(arr_sq)
print(arr_sq.shape)
#[5 6 7 8 9]
#(5,)

sd = nm.arange(25).reshape(5,5)
print(sd)
#[[ 0  1  2  3  4]
# [ 5  6  7  8  9]
# [10 11 12 13 14]
# [15 16 17 18 19]
# [20 21 22 23 24]]

sdx = sd[1:2]
print(sdx)
#[[5 6 7 8 9]]
sdx[0,0] = 100
print(sdx)
#[[100   6   7   8   9]]
print(sd)
#[[  0   1   2   3   4]
# [100   6   7   8   9]
# [ 10  11  12  13  14]
# [ 15  16  17  18  19]
# [ 20  21  22  23  24]]
#here we created an array sd, then created sdx using sd. After that we change the value of sdx to have
#100, then when we print our original sd, we see the changes reflected. Slice operator which returns is 
#a reference to sd original array.

print(sd[::2,[1,3,4]])
#here take all the rows from 0 till 4 and skip by 2, and for coulmn take 1,3 and 4 
#[[ 1  3  4]
# [11 13 14]
# [21 23 24]]

#here ix_ creates the index for the data to be retrieved from the array, 
#we get follwoing row cols combination (0,0),(0,2),(0,3),(2,0),(2,2) and (2,3).
abc = nm.ix_([0,2],[0,2,3])
print(sd[abc])
#[[ 0  2  3]
# [10 12 13]]

#Exercise to add two array, aq and bq
aq =nm.arange(25).reshape((5,5))
bq =nm.arange(75).reshape((5,5,3))

aq_new = aq[:,:,nm.newaxis]
final_ans = aq_new + bq
print('*****')
print(final_ans)
print(final_ans[::2,:])

an = nm.array([1,2,3,4,5])
print(an.ndim)
print(an.dtype)
print(an.itemsize)
print(an.nbytes)
an[0] = 100.34
print(an)
#We are moving 100.34 into first position if the array, butwe are getting :
#output as [100   2   3   4   5]. since 'an' is a numeric type, so the decimal part is truncated 
#Here it gives the total number of bytes used, its '20' here.
#This itemsize is '4' which gives byte per element
#ndim gives us the dimention of the array, here its '1'.
#dtype will give the integer type, int32
    

ap = nm.array([1,2,3,4.56,5])
print(ap.dtype)
#float64
#Since there is one float value the numpy will take the dtype as 'float64'.

#Stride trick to make a sliding window out of an array
mm = nm.arange(10)
s=2
w=4
print(mm)
#[0 1 2 3 4 5 6 7 8 9]
print(nm.lib.stride_tricks.as_strided(mm, shape=(len(mm)-w+1,w),strides=mm.strides * 2))

