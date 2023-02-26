df.isnull().sum() # смотрим сумму NaN для каждой колонки в df

df.drop('county_name',   axis='columns', inplace=True)# Тупо сбрасываем колонку 'country_name' в df и делаем это все внутри самой функции 

ri.dropna(subset=['stop_date', 'stop_time'], inplace=True)# Cбрасываем NaN в колонках,делая это все в самой таблице 

df.dtypes # смотрим,какие типы у нас есть 
# object : Python strings(or other Python objects)
# bool
# int , float , datetime , category(category : uses less memory and runs faster)

apple['price'] =   apple.price.astype('float')# переводим колонку 'price' во float 

apple.date.str.replace('/', '-') # заменяем в колонке date / на -

apple.date.str.cat(apple.time, sep=' ') # соединяем две колонки в таблице с разделителем ''

apple['date_and_time'] = pd.to_datetime(combined)# переводим весь код во время 

apple.set_index('date_and_time', inplace=True) # ставим индекс

ri.stop_outcome.value_counts(normalize = True) # считаем количество значений(отличающихся) для колонки + добавляем normalize,что считает количество процентов 

# оператор and в действии
female_and_arrested = ri[(ri.driver_gender == 'F') &
                         (ri.is_arrested == True)]

# оператор or в действии 
female_or_arrested = ri[(ri.driver_gender == 'F') |
                        (ri.is_arrested == True)]

# Если мы считаем True and False then True = 1, False = 0

df.district.unique()# смотрим уникальные значения district

ri.groupby('district').is_arrested.mean()# группировка district по is_arrested.mean() для каждого district 

ri.groupby(['driver_gender', 'district']).is_arrested.mean()# группируем по двум значениям 

ri.search_type.value_counts(dropna=False)# считаем значение и NaN 

ri['inventory'] = ri.search_type.str.contains('Inventory', na=False) # штука,которая ищет значение "Inventory" по всей колонке(Если находит,то True) и если NaN,то возвращает False и добавляет это все в новую колонку 

# Плотим BAR на оси y,то есть вертикальный bar 
search_rate.sort_values().plot(kind='barh')
plt.show()


apple.date_and_time.dt.month # cмотрим месяцы,если колонка уже переведена во время 

apple.set_index('date_and_time', inplace=True) # добавляем индекс 

apple.index.month # смотрим месяц из индекса 

monthly_price = apple.groupby(apple.index.month).price.mean()# смотрим среднюю цену за каждый месяц 

apple.price.resample('M').mean()# смотрим цены по каждому месяцу 

# объединение двух таблиц по колонке месяца
monthly_price = apple.price.resample('M').mean()
monthly_volume = apple.volume.resample('M').mean()
pd.concat([monthly_price, monthly_volume], axis='columns')

monthly.plot(subplots=True) # subplots создают два графика вместо одного сложного(иногда полезно)

pd.crosstab(ri.driver_race, ri.driver_gender) # создаем таблицу из двух значений driver_race-index,driver_gender - columns

# метод loc
table.loc['Asian':'Hispanic']

table.plot(kind='bar', stacked=True)# одно значение позади другого

# Используем map(mapping),чтобы заменить на те значения,которые у нас в dict
mapping = {'up':True, 'down':False}
apple['is_up'] = apple.change.map(mapping)

weather[['AWND', 'WSF2']].describe()#так тоже можно дескрайбить 

temp = weather.loc[:, 'TAVG':'TMAX'] #берем все строки и колонки от 'TAVG' do  'TMAX'

ri.stop_length.memory_usage(deep=True)# смотрим,сколько у нас весит колонка 


cats = ['short', 'medium', 'long']# создаем последовательность + категории
ri['stop_length'] = ri.stop_length.astype('category',ordered=True,categories=cats)# astyp им
ri.stop_length.memory_usage(deep=True) # смотрим,сколько памяти занимает,сразу можно сказать,что меньше


# объединяем две таблицы по date и DATE левым объединением 
apple_high = pd.merge(left=apple, right=high,
                      left_on='date', right_on='DATE',                      
                      how='left')

search_rate.loc['Equipment', 'M'] #выбираем двойной,сложный индекс.Сначала у нас идет значение индекса , потом пол 

search_rate.unstack() # c помощью unstack() мы можем из двойного индекса в таблице перейти к простой таблице с одним индексом  

# pivot_table - тоже самое,что и unstack(),только позволяет выставлять еще и индексы,колонки и значение(более гибкий метод)
ri.pivot_table(index='violation',
               columns='driver_gender',               
               values='search_conducted')


#«Правило квадратного корня» — это обычно используемое эмпирическое правило для выбора количества бинов: выберите количество бинов как квадратный корень из числа выборок. 
