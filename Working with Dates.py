# Import date 
from datetime import date #импортируем библиотеку для создания дат 
# Create dates
two_hurricanes_dates = [date(2016, 10, 7), date(2017, 6, 21)]
print(two_hurricanes_dates[0].year)
print(two_hurricanes_dates[0].month)
print(two_hurricanes_dates[0].day)

print(two_hurricanes_dates[0].weekday())# узнаем день недели(отсчет начинается от 0

# Минусуем даты
delta = d2 - d1
print(delta.days)

# Import timedelta
from datetime import timedelta
# Create a 29 day timedelta
td = timedelta(days=29) #timedelta - промежуток времени,который мы создаем 


x += 1 #более красивый способ добавить 1


# Express the date in ISO 8601 format 
df.isoformat()

print(sorted(some_dates))# ISO 8601 можно легко сортировать 

d.strftime("%Y" или "Year is %Y" или "%Y/%m/%d") #помогает изменять дату в зависимости от значений,перевод из строки во время 
-----------------------------------------------------------------------------------------------------------------------
# Import datetime
from datetime import datetime
dt = datetime(2017, 10, 1, 15, 23, 25, 500000)# добавляем точность до минут,секунд,часов, миллисекунды 

from datetime import datetime
dt = datetime(year=2017, month=10, day=1,hour=15, minute=23, second=25,microsecond=500000) #так тоже можно 

dt_hr = dt.replace(minute=0, second=0, microsecond=0)# меняем некоторые значения 

dt = datetime.strptime("12/30/2017 15:19:13", "%m/%d/%Y %H:%M:%S" или "%Y-%m-%d %H:%M:%S" или "%Y-%m-%d" или "%H:%M:%S on %Y/%m/%d/")#переводим строку во время 

---------
ts = 1514665153.0
print(datetime.fromtimestamp(ts))# переводим время из timestapm(1970,которая всегда выражается в секундах от момента создания) в нормальное время
2017-12-30 15:19:13

print(duration.total_seconds())#принтим полный промежуток времени в секундах(так же есть вариации,в чем)

td = timedelta(days=29, seconds = 1, weeks = 5)


