# Берем случайные 10 строк из таблицы(также можно сделать и с колонкой)
pts_vs_flavor_samp = pts_vs_flavor_pop.sample(n=10)

cup_points_samp = coffee_ratings['total_cup_points'].sample(n=1

# Плотим хистограмму с шагом - 2
import matplotlib.pyplot as plt
import numpy as np
coffee_ratings["total_cup_points"].hist(bins=np.arange(59, 93, 2))plt.show()

# Можно брать рандомные значения по разным законам 
randoms = np.random.beta(a=2, b=2, size=5000)

np.random.normal(loc=2, scale=1.5, size=2)

# Берем 5 любых строчек,random_state -> seed
coffee_ratings.sample(n=5, random_state=19000113)


# Деление без остатка 
interval = pop_size // sample_size

# Берем первые 267 строчек 
interval = 267 
coffee_ratings.iloc[::interval]

# Перемешиваем все строки 
shuffled = coffee_ratings.sample(frac=1)
shuffled = shuffled.reset_index(drop=True).reset_index() # Два раза резетим индекс,чтобы восстановить индексацию от 1 до ....
shuffled.plot(x="index", y="aftertaste", kind="scatter")
plt.show()


# Группируем по 'country_of_origin' и после этого берем 10% всех данных
coffee_ratings_strat = coffee_ratings_top.groupby("country_of_origin").sample(frac=0.1, random_state=2021)

# Берем с каждой 'country_of_origin' по 15 строк 
coffee_ratings_eq = coffee_ratings_top.groupby("country_of_origin").sample(n=15, random_state=2021))

# Берем 3 любых значения 'variety',после этого чистим таблицу от ненужных значений и берем 5 семплов с каджого 'variety'
varieties_pop = list(coffee_ratings['variety'].unique())
import random
varieties_samp = random.sample(varieties_pop, k=3)
variety_condition = coffee_ratings['variety'].isin(varieties_samp)
coffee_ratings_cluster = coffee_ratings[variety_condition]
coffee_ratings_cluster.groupby("variety").sample(n=5, random_state=2021)

# Берем треть всего датафрейма(берем рандомно)
coffee_ratings_srs = coffee_ratings_top.sample(frac=1/3, random_state=2021)

# Код,который помогает найти все возможные значения(перемешанные)
dice = expand_grid(  {  'die1': [1, 2, 3, 4, 5, 6],
                        'die2': [1, 2, 3, 4, 5, 6],
                        'die3': [1, 2, 3, 4, 5, 6],
                        'die4': [1, 2, 3, 4, 5, 6]  })

# Если у нас DataFrame,то мы используем df.plot()
# Если у нас list или np.array,то используем plt.plot(x = x,y = y,data = data )

# Используем ddof = 0,когда мы говорим о всем DataFrame,ddof = 1,когда о выборке(связано со степениями свободы)
coffee_ratings['total_cup_points'].std(ddof=0)

# replace = True показывает,что когда мы берем данные из датафрейма,то,после этого они становятся на место и теоретически возможно,
# что мы их можем вытянуть еще раз
coffee_resamp = coffee_focus.sample(frac=1, replace=True)

# Gausan distribution == Normal Distribution 

# std(всей колонки) = std(выборки) * np.sqrt(len(выборки))


standard_error = np.std(bootstrap_distn, ddof=1)
std = standard_error * np.sqrt(500)
assert coffee_sample['flavor'].std() = std  

# Вычисляем доверительный интервал(97 %)
np.quantile(coffee_boot_distn, 0.025)
np.quantile(coffee_boot_distn, 0.975)
# Потом я бы вычислил медиану 
