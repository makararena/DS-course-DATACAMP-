filename = 'huck_finn.txt'
file = open(filename, mode='r')  # 'r' is to read,открываем 2.0
text = file.read()#выводит сам файл 
file.close()#закрывает файл 
# нужно просто для ознакомления с данными 

filename = 'huck_finn.txt'
file = open(filename, mode='w')  # 'w' is to write
file.close()
#нужно,чтобы просто записать что-то в файл

with open('huck_finn.txt', 'r') as file:
        print(file.read())
# легче использовать именно этот код и тогда не придеться потом все закрывать 

print(dt.readline())
#выводит только одну строчку из всего кода 

------
#плоские файлы-файлы(.txt,.csv) состоят из чисел или строк с числами....данные разделяются разделителями,таюличные значения
#также имеют заголовки и очень важно знать,есть ли заголовок 
# c плоскими данными легче всего работать через np или pandas
------
import numpy as np
filename = 'MNIST_header.txt'
data = np.loadtxt(filename,# берем сам файл
                  delimiter=',',# разделитель(нужен явный)
                  skiprows=1,# сколько строк мы пропускаем 
                  usecols=[0, 2])# какие колонки мы используем 
                  dtype=str # во что мы преобразуем все колонки(тип)
print(data)
#просто загружаем массив данных

data = np.genfromtxt('titanic.csv',
                      delimiter=',', 
                      names=True,#Указываем,есть ли заголовки 
                      dtype=None)#Даем np самому указать тип файлов 
# создает интуитивно понятную строку из всех значений,также можно забирать сразу весь столбец df['столбец']

np.recfromtxt()
# работает также,как и np.genfromtx,только она более простая и все значения верхней функции по дефолту и оба подходят для импорта смешанных данных


----
# Assign the filename: file
file = 'digits.csv'

data = pd.read_csv(file,
                   nrows = 5,# Read the first 5 rows
                   header = None #убираем заголовок 
                   sep='\t' # разделяем пробелом 
                   comment = '#' # убираем все комментарии,которые идут после знака '#'
                   na_values = 'Nothing' # значения,которых няма переводим в 'NaN'

# Build a numpy array from the DataFrame: data_array
data_array = np.array(data)
# импортируем данные как pandas и переводим в листы 


import os
wd = os.getcwd()
os.listdir(wd)
# также выводит каталог файлов,которые сейчас находятся в оболочке 
-----
import pickle 
with open('pickled_fruit.pkl', 'rb')) as file:
            data = pickle.load(file)    
print(data)
# пиклы нужны,потому что внутри них иерархическая система,'rb'-сохраняется двоичная система(key:value)-преобразуется в словарь
----
import pandas as pd
file = 'battledeath.xlsx'
xls = pd.ExcelFile(file)
print(xls.sheet_names)
df1 = data.parse('1960-1966') # sheet name, as a string
df2 = data.parse(0) # sheet index, as a float
df3 = xls.parse(1, usecols=[0], skiprows=[0], names=['Country'])#берем второй лист таблицы,первую колонку,пропускаем первую строку и называем колонку 'Country'

# импорт названий листов из Exel и потом уже присваивание к переменным отдельные листы 

-----

from sas7bdat import SAS7BDAT

with SAS7BDAT('sales.sas7bdat') as file:
    df_sas = file.to_data_frame()

print(df_sas.head())
#импорт SAS и перевод в pd.DataFrame      SAS разработан для бизнеса 
------
import pandas as pd
df = pd.read_stata('disarea.dta')
print(df.head())
# импортирование Stata используется для решения статистических задач 
------
import h5py 
file = 'LIGO_data.hdf5'
data = h5py.File(file, 'r')#делаем так,чтобы мы могли просто прочитать это файл,обязательно в конце надо будет закрыть его 
print(type(data))

# Принтим ключи этого формата(вроде должно быть 3)
for key in data.keys():
    print(key)
# HDF5 нужен для сохранение большого количества информации и масштабирование его 

print(data['meta']['Description'].value)
# залезаем во внутреннюю группу группы meta
------
import scipy.io
mat = scipy.io.loadmat('albeck_gene_expression.mat')
#импорт MATLAB файлов просто вычислительные программы,от которых общество еще не успело отказаться 
print(mat.keys())#принтим ключи 
print(type(mat['CYratioCyt']))#узнаем тип ключей
print(np.shape(mat['CYratioCyt']))# узнаем тип ключа


# Существует такая штука,как Relationalmodel то есть нам даны как минимум две таблицы и они связанны какой-то колонкой,которая у них одинаковая,12 заповидей и тп 
# для взятия данных из них нам понадобятся SQL запросы sqllite,postgreSQL и MySQL

from sqlalchemy import create_engine # импортим sqlalchemy(лучше всего работает со вмести базами данных)
engine = create_engine('sqlite:///Northwind.sqlite')#создаем механизм 
table_names = engine.table_names()#сохраняем имена таблиц 
print(table_names)


from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Northwind.sqlite')#создаем механизм 
con = engine.connect()#создаем соединение 
rs = con.execute("SELECT * FROM Orders")#cоздаем запрос 
df = pd.DataFrame(rs.fetchall())#берем все колонки из запроса 
df.columns = rs.keys()#добавляем названия колонкам 
con.close()#закрываем соединение 
# выборка и потом сохранение в pd


from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Northwind.sqlite')
with engine.connect() as con:
        rs = con.execute("SELECT OrderID, OrderDate, ShipName FROM Orders")
        df = pd.DataFrame(rs.fetchmany(size=5))    
        df.columns = rs.keys()
# тоже самое,только нам не придеться закрывать соединение 

con.execute("SELECT * FROM Customer ORDER BY SupportRepId")
# функция ORDER BY помогает добавить порядок 

("SELECT * FROM Customer WHERE Country = 'Canada'")
# помогает добавить условие 

------------------------------------
# Import packages
from sqlalchemy import create_engine
import pandas as pd

# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')

# Execute query and store records in DataFrame: df
df = pd.read_sql_query("SELECT * FROM Album", engine)
#Выводит тоже самое,просто в одну строчку 


("SELECT OrderID, CompanyName FROM Orders INNER JOIN Customers on Orders.CustomerID = Customers.CustomerID")
# соединение между двумя RelationalTables
----------------------
from urllib.request import urlretrieve 
import pandas as pd
url = 'https://s3.amazonaws.com/assets.datacamp.com/production/course_1606/datasets/winequality-red.csv'
urlretrieve(url,'winequality-red.csv')
df = pd.read_csv('winequality-red.csv', sep=';')
print(df.head())
# импортируем данные,сначала сохраняя их локально,потом перезаписываем в df и используем 
--------------------
import matplotlib.pyplot as plt
import pandas as pd
url = 'https://s3.amazonaws.com/assets.datacamp.com/production/course_1606/datasets/winequality-red.csv'
df = pd.read_csv(url,sep = ';')
print(df.head())
# просто принтим,пока без сохранения 

xls = pd.read_excel(url,sheet_name=None)
# используем абсолютно все листы в exel

-------
# импорт HTTP request 
from urllib.request import urlopen, Request
url = "https://campus.datacamp.com/courses/1606/4135?ex=2"
request = Request(url)
response = urlopen(request)
html = response.read()
print(html)
response.close()
#старенький,но рабочий метод 

import requests 
url = "http://www.datacamp.com/teach/documentation"
r = requests.get(url)
text = r.text
print(text)
#более продвинутый способ 
----------------------------------------------------------
# Import packages
import requests
from bs4 import BeautifulSoup # импортируем 
url = 'https://www.python.org/~guido/'
r = requests.get(url)#делаем запрос продвинутым способом 
html_doc = r.text #читаем текст 
soup = BeautifulSoup(html_doc)#просто используем красивый суп,чтобы перевести в другой тип(pretty soup),где уже работает prettify()
pretty_soup = soup.prettify()#улучшаем 
guido_title = soup.title #сохраняем заголовок 
guido_text = soup.get_text() #сохраняем сам текст
print(pretty_soup)
# bs4 - это штука,которая делает код более красивым 
--
a_tags = soup.find_all('a')#ищем все гипперсслки 

# Print the URLs to the shell
for link in a_tags:
    print(link.get('href'))#перечесляем все гипперссылки 


----------------------------
    # Load JSON: json_data
with open("a_movie.json") as json_file:
    json_data = json.load(json_file)# загружаем json
    

# Print each key-value pair in json_data
for k in json_data.keys():
    print(k + ': ', json_data[k])#перечесляем наш словарь 
#json нужен для API(скорее всего импортируемые данные будут именно в этом формате )
--------------------------
import requests
url = 'https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&exintro=&titles=pizza'
r = requests.get(url)
json_data = r.json()
pizza_extract = json_data['query']['pages']['24768']['extract']
print(pizza_extract)
# создание запроса по API и получение данных
