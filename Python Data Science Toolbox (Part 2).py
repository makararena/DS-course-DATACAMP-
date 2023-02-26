# Iterable
#   *Examples:lists,strings,dictionaries,file connections
#    *An object with an associated iter() method 
#    *Applying iter() to an iterable creates an iterator 
# Iterator 
#   Produces next value with next()
------------------------------------------------------------------------------------------
word = 'Da'
it = iter(word)
next(it)

'D'

next(it)

'a'
# итерация итератора 
------------------------------------------------------------------------------------------

word = 'Data'
it = iter(word)
print(*it)
# полная итерация и итератора 
------------------------------------------------------------------------------------------

file = open('file.txt')
it = iter(file)
print(next(it))
# итерация файла 

------------------------------------------------------------------------------------------

avengers = ['hawkeye', 'iron man', 'thor', 'quicksilver']
e = enumerate(avengers)# создает индексы по словам 
print(type(e))

<class 'enumerate'>

e_list = list(e)
print(e_list)
[(0, 'hawkeye'), (1, 'iron man'), (2, 'thor'), (3, 'quicksilver')]

for index, value in enumerate(avengers, start=10):
        print(index, value)
# можно указать с какого индекса будет начинаться индексация 
------------------------------------------------------------------------------------------

avengers = ['hawkeye', 'iron man', 'thor', 'quicksilver']
names = ['barton', 'stark', 'odinson', 'maximoff']
z = zip(avengers, names)
print(type(z))
# zip склеивает два листа 
#[('hawkeye', 'barton'), ('iron man', 'stark'), ('thor', 'odinson'), ('quicksilver', 'maximoff')]
------------------------------------------------------------------------------------------
avengers = ['hawkeye', 'iron man', 'thor', 'quicksilver']
names = ['barton', 'stark', 'odinson', 'maximoff']
for z1, z2 in zip(avengers, names):
        print(z1, z2)
#создание четырех tuple 


import pandas as pd 
result = []
for chunk in pd.read_csv('data.csv', chunksize=1000):
        result.append(sum(chunk['x']))
        total = sum(result)
        print(total)
#создание корзин с определенном количеством данных в них


new_nums = [num + 1 for num in nums]
# синтаксис для кодирования в одну строчку 

pairs_2 = [(num1, num2) for num1 in range(0, 2) for num2 in range(6, 8)]
print(pairs_2)
#создание листа из туплов ДВОЙНОГО 


[num ** 2 if num % 2 == 0 else 0 for num in range(10)]
# создание условий в одну строчку кода 

pos_neg = {num: -num for num in range(9)}
print(pos_neg)
# создание словарей, num-key , -num это value 

(2 * num for num in range(10))
# создание генераторов,почти тоже самое,но они не хранят у себя в памяти значения(как листы),а только код

def num_sequence(n):
    """Generate values from 0 to n."""
        i = 0 
        while i < n:
            yield i
            i += 1
# функция по генератору,тоже самое,только использует yield и не держит у себя никаких данных
for item in result:
        print(item)
0
1
2
3
4
# вывод

