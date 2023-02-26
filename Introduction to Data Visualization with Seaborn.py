import seaborn as sns
import matplotlib.pyplot as plt
sns.relplot(x="total_bill",
            y="tip",             
            data=tips,            
            kind="scatter",            
            col="day"# создание дополнительной колонки,            
            row="time"#создание строчек с дополнительным указанием smoker or not)
            col_wrap=2 # после каждых двух новых графиков новая строка 
            col_order=["Thur","Fri","Sat","Sun"])
            size="size" #размер по данным на граффике 
            hue='size' # палира тоже может изменяться по цвету 
            style="smoker" #добавление знаков на отделные точки 
            alpha=0.4 # добавление прозрасности 

            

plt.show()
#использование relpot() для созданя графика 



import matplotlib.pyplot as plt
import seaborn as sns
sns.relplot( x="hour",
             y="NO_2_mean",
             data=air_df_loc_mean,             
             kind="line",             
             style="location", # добавление стиля            
             hue="location", # добавление расцветки            
             markers=True,# добавление маркеров             
             dashes=False # бираются '--',которые появились из-за стиля
             ci="sd" #замена доверительного интервала(95%) среднеквадратическим отклонением)
             ci=None #просто все убираем 
plt.show()
# использование relplot()


import matplotlib.pyplot as plt
import seaborn as sns
category_order = ["No answer", "Not at all","Not very", "Somewhat", "Very"]
sns.catplot(x="how_masculine",
            data=masculinity_data,
            kind="count",
            order=category_order)
plt.show()
#использование countplot,похож на гистограму 

import matplotlib.pyplot as plt
import seaborn as sns
sns.catplot(x="day",
            y="total_bill",            
            data=tips,            
            kind="bar",            
            ci=None)
plt.show()
# создание бара,также похож на гистограмму,также убираем доверительные интервалы 

import matplotlib.pyplot as plt
import seaborn as sns
g = sns.catplot(x="time",
                 y="total_bill",
                data=tips,
                kind="box",
                sym=""#убираем разброс
                whis=2.0/whis=[5, 95]/whis=[0, 100]#редактирование усов 1-IQR,2-5,95 процентиль,3-все)

plt.show()
# создание ящика с усами

import matplotlib.pyplot as plt
import seaborn as sns
sns.catplot(x="age",
            y="masculinity_important",
            data=masculinity_data,
            hue="feel_masculine",
            kind="point",            
            join=False# убираем соединение между точками
            estimator=median #добавление медианы вместо медианы
            capsize=0.2 #увеличение 'усов'
            ci=None # просто все убираем)
plt.show()
# создание графика с 'треугольниками' 

sns.set_style("whitegrid")#добавление другого стиля 

custom_palette = ['#FBB4AE', '#B3CDE3', '#CCEBC5', '#DECBE4', '#FED9A6', '#FFFFCC', '#E5D8BD', '#FDDAEC', '#F2F2F2']
sns.set_palette()#создание и добавление другой палитры 
sns.set_context("paper","notebook","talk","poster")#изменение размера 

g.fig.suptitle("New Title",y=1.03)#добавление названия

g.set_titles("This is {col_name}")#добавление названия для двух групп сразу

g.set(xlabel="New X Label",       
      ylabel="New Y Label")#добавление названий по оси x и y

plt.xticks(rotation=90)#поворот стиков на 90 градусов 

plt.clf()#удаляет старые графики,не позволяя им скапливаться 
