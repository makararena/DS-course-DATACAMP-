df.dtypes # показывает типы(str,int,object) колоночек 
sales['Revenue'] = sales['Revenue'].str.strip('$')#убираем знак доллара со всех строк колонки 
sales['Revenue'] = sales['Revenue'].astype('int') #переводит с типа category в int
assert sales['Revenue'].dtype == 'int' #проверяем,какой тип у нас получился(если ничего не возращает,то True)
#str int float bool datetime category - типы данных 

import datetime as dt #импорт библиотеки 
today_date = dt.date.today() # назначение сегодняшней даты 
user_signups[user_signups['subscription_date'] > today_date]
# извлекаем те данные,которые больше,чем сегодня 

---------# как можно бороться с потерями или ошибками данных,если они числовые(потери)
# Drop values using filtering
movies = movies[movies['avg_rating'] <= 5] 
# Drop values using .drop() 
movies.drop(movies[movies['avg_rating'] > 5].index, inplace = True)

# Assert results 
assert movies['avg_rating'].max() <= 5
# Convert avg_rating > 5 to 5

movies.loc[movies['avg_rating'] > 5, 'avg_rating'] = 5
---------# как можно бороться с потерями или ошибками данных,если они ДАТЫ(потери)
import datetime as dt 
user_signups['subscription_date'] = pd.to_datetime(user_signups['subscription_date']).dt.date
# как переводить время(str) в тип(время)
# Drop values using filtering
user_signups = user_signups[user_signups['subscription_date'] < today_date]
# Drop values using .drop()
user_signups.drop(user_signups[user_signups['subscription_date'] > today_date].index, inplace = True)
# Drop values using filtering
user_signups.loc[user_signups['subscription_date'] > today_date, 'subscription_date'] = today_date
# Assert is true
assert user_signups.subscription_date.max().date() <= today_date


duplicates = height_weight.duplicated()
height_weight[duplicates]
#смотрим,какие есть дубликаты
-------
#subset : List of column names to check for duplication
#keep : Whether to keep first('first') , last('last') or all(False) duplicate values
column_names = ['first_name','last_name','address']
duplicates = height_weight.duplicated(subset = column_names, keep = False)

------#Treating dublicates
# Drop complete duplicates from ride_sharing
ride_dup = ride_sharing.drop_duplicates()
# Create statistics dictionary for aggregation function
statistics = {'user_birth_year': 'min', 'duration': 'mean'}
# Group by ride_id and compute new statistics
ride_unique = ride_dup.groupby('ride_id').agg(statistics).reset_index()
# Find duplicated values again
duplicates = ride_unique.duplicated(subset = 'ride_id', keep = False)
duplicated_rides = ride_unique[duplicates == True]
# Assert duplicates are processed
assert duplicated_rides.shape[0] == 0

----------------------------------------------------------------------------------------------------------------------------------------
inconsistent_categories = set(study_data['blood_type']).difference(categories['blood_type'])#находим те значения,которых нету в categories
inconsistent_rows = study_data['blood_type'].isin(inconsistent_categories)
inconsistent_data = study_data[inconsistent_rows]
# Drop inconsistent categories and get consistent data only
consistent_data = study_data[~inconsistent_rows]# ~ означает 'все кроме' 
print(inconsistent_categories)#left anti join
#categories-просто таблиц с возможными значениями 
#study data-наша таблица 


marriage_status['marriage_status'] = marriage_status['marriage_status'].str.upper()
#делаем все строки в высоком регистре 
marriage_status['marriage_status'] = marriage_status['marriage_status'].str.lower()
#делаем все строки в нижнем регистре 
demographics = demographics['marriage_status'].str.strip()
# Удаляем все пробелы 
------------
# Using cut() - create category ranges and names
ranges = [0,200000,500000,np.inf]
group_names = ['0-200K', '200K-500K', '500K+']
# Create income group column
demographics['income_group'] = pd.cut(demographics['household_income'], bins=ranges,
                                       labels=group_names)
demographics[['income_group', 'household_income']


# Create mapping dictionary and replace 
mapping = {'Microsoft':'DesktopOS', 'MacOS':'DesktopOS', 'Linux':'DesktopOS','IOS':'MobileOS', 'Android':'MobileOS'}
devices['operating_system'] = devices['operating_system'].replace(mapping)
devices['operating_system'].unique()


phones["Phone number"] = phones["Phone number"].str.replace("+", "00") #заменяем + на нули

phones.loc[digits < 10, "Phone number"] = np.nan #те,которые меньше 10,те Nan


# Find length of each row in Phone number column
sanity_check = phone['Phone number'].str.len()

# Assert minmum phone number length is 10
assert sanity_check.min() >= 10

# Assert all numbers do not have "+" or "-"
assert phone['Phone number'].str.contains("+|-").any() == False

# Replace letters with nothing
phones['Phone number'] = phones['Phone number'].str.replace(r'\D+', '')#заменяем любые вещи,кроме цифр 
phones.head()

-----------------------------------------------------------------------------------------------------------------------------------
# как исправлять только отдельные вещи в данных
temp_fah = temperatures.loc[temperatures['Temperature'] > 40, 'Temperature']#сначала берем температуру больше 40 и перезаписываем в переменную
temp_cels = (temp_fah - 32) * (5/9)# потом вычесляем по формуле нормальные значения 
temperatures.loc[temperatures['Temperature'] > 40, 'Temperature'] = temp_cels # перезаписываем данные уже в изначальную таблицу 

-----
birthdays['Birthday'] = pd.to_datetime(birthdays['Birthday'],
                                        # пробует использовать разные форматы,чтобы перевести их в нормальный вид дат 
                                        infer_datetime_format=True,
                                        # Return NA for rows where перевод failed
                                        errors = 'coerce')

birthdays['Birthday'] = birthdays['Birthday'].dt.strftime("%d-%m-%Y")#перевод всего формата в другой формат dt.strfttime()


---------#Называется Cross field validation
sum_classes = flights[['economy_class', 'business_class', 'first_class']].sum(axis = 1)#сначала вычесляем сумму всех классов в таблице 
passenger_equ = sum_classes == flights['total_passengers'] #сравниваем с итогом,который также был в таблице и плучаем тех,True или False 
# Find and filter out rows with inconsistent passenger totals
inconsistent_pass = flights[~passenger_equ]# берем все False 
consistent_pass = flights[passenger_equ]# берем все True 


---#Тоже самое,только со временем 
import pandas as pd
import datetime as dt
# Convert to datetime and get today's date
users['Birthday'] = pd.to_datetime(users['Birthday'])
today = dt.date.today()
# For each row in the Birthday column, calculate year difference
age_manual = today.year - users['Birthday'].dt.year
# Find instances where ages match
age_equ = age_manual == users['Age']
# Find and filter out rows with inconsistent age
inconsistent_age = users[~age_equ]
consistent_age = users[age_equ]


---#Можем посмотреть по таблице значений,которых нет 
df.isna()
airquality.isna()#там уже добавляется сумма по столбцам 


import missingno as msno
import matplotlib.pyplot as plt
msno.matrix(airquality)
plt.show()
#как визуализировать потери с помощью графика 


----------
~ - не 
----------
missing = airquality[airquality['CO2'].isna()]#данные,котоые c NaN или NaT
complete = airquality[~airquality['CO2'].isna()]#данные,которые у нас есть 

airquality.sort_values(by = 'Temperature')#сортим датафрейм 

#MCAR MAR MNAR

airquality_dropped = airquality.dropna(subset = ['CO2'])#Dropping missing values 

co2_mean = airquality['CO2'].mean()
airquality_imputed = airquality.fillna({'CO2': co2_mean})#replacing with statistical measurments 


---------------------------------------------------------------
from fuzzywuzzy import fuzz
# Сравнивание строк на схожесть 
fuzz.WRatio('Reeding', 'Reading')
----
# изначально бло два списка(categoties-со всеми нормальными категориями) и survey(там уже говнячие данные)
# For each correct category
for state in categories['state']:
    # Find potential matches in states with typoes    
    matches = process.extract(state, survey['state'], limit = survey.shape[0])# каждый нормальный штат сравниваем по схожести с опросом + лимит в длинну опроса 
    # For each potential match match
    for potential_match in matches:
        # If high similarity score
        if potential_match[1] >= 80:    #если рейтинг потенциала(потенциал второй по счету),то заменяем штат с ошибкой на нормальный штат 
            # Replace typo with correct category
            survey.loc[survey['state'] == potential_match[0], 'state'] = state


shape[n,m] ; n-columns,m-rows

print(survey['state'].unique())#уникальные значения графиков 


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Record linkage is a powerful technique used to merge multiple datasets together,
#used when values have typos or different spellings. In this chapter, 
#you'll learn how to link records by calculating the similarity between strings

#Import recordlinkage and generate full pairs
import recordlinkage
indexer = recordlinkage.Index() #создаем объект индексации(нужен для создания самих пар)
indexer.block('state')#берем столбец state,как объект для индексации,чтобы с помощью него уже создавать пары 
full_pairs = indexer.index(census_A, census_B)#уже создаем сами пары с индексом state и получаем список из всех возможных соеденений по штатам 

# Comparison step
compare_cl = recordlinkage.Compare() #создаем функцию сравнения 
compare_cl.exact('date_of_birth', 'date_of_birth', label='date_of_birth')# берем те,которые должны совпадать 100%
compare_cl.exact('state', 'state', label='state')# тоже самое 
compare_cl.string('surname', 'surname', threshold=0.85, label='surname')#вычисляем сходство строк(берем только те,которые нам нужны)
compare_cl.string('address_1', 'address_1', threshold=0.85, label='address_1')#тоже самое 

potential_matches = compare_cl.compute(full_pairs, census_A, census_B)# вычисляем совпадения при помощи метода compute,также важна последовательность census_A , census_B

# Import recordlinkage and generate pairs and compare across columns...


# Isolate matches with matching values for 3 or more columns
matches = potential_matches[potential_matches.sum(axis = 1) >= 3]# вычисляем нужную нам сумму потенциалных совпадений среди всех строк 
# Get index for matching census_B rows only
duplicate_rows = matches.index.get_level_values(1)#получаем все индексы B(1 - потому что это второй(внутренний индекс))
# Finding new rows in census_B
census_B_new = census_B[~census_B.index.isin(duplicate_rows)]# берем все индексы,которых не было,когда мы связывали 2 таблицы,потому что брали именно данные,которые сходятся.
# и при этом есть в обычной таблице т.е просто берем те данные,которых не было в dublicate_rows
# Link the DataFrames!
full_census = census_A.append(census_B_new)#все склеиваем и получаем таблицу,в которой значения не повторяются 
