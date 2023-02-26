# Вычисляем теоретическую CDF 
import numpy as np
import matplotlib.pyplot as plt
mean = np.mean(michelson_speed_of_light)
std = np.std(michelson_speed_of_light)
samples = np.random.normal(mean, std, size=10000)


# создание линейной регрессии
a, b = np.polyfit(illiteracy,fertility,1)
# наклон 
print('slope =', a, 'children per woman / percent illiterate')
# x
print('intercept =', b, 'children per woman')
# Make theoretical line to plot
x = np.array([0,100])
y = a * x + b
# Add regression line to your plot
_ = plt.plot(x, y)
plt.xlabel('percent illiterate')
plt.ylabel('fertility')
# Draw the plot
plt.show()

# создает array в 200 знаков от 0 до 0.1
a_vals = np.linspace(0, 0.1, 200)



# Compute sum of square of residuals for each value of a_vals
for i, a in enumerate(a_vals):
    rss[i] = np.sum((fertility - a*illiteracy - b)**2)

# создает(инициализирует) array
rss = np.empty_like(a_vals)

# красиво распаковывает файлы
for x, y in zip(anscombe_x , anscombe_y):


# рандомно берем числа из array 
import numpy as np
np.random.choice([1,2,3,4,5], size=5)

# создание похожей функции
def bootstrap_replicate_1d(data, func):
    """Generate bootstrap replicate of 1D data."""    
    bs_sample = np.random.choice(data, len(data))
    return func(bs_sample)

# ПОВТОРЕНИЕ 10 000 раз тоже самое 
bs_replicates = np.empty(10000)
for i inrange(10000):
        bs_replicates[i] = bootstrap_replicate_1d(michelson_speed_of_light, np.mean)


# вычисление 95% доверительного интервала 
conf_int = np.percentile(bs_replicates, [2.5, 97.5])


----------------------------------------
# Весь этот код создает сразу несколько линий регрессий по закону np.random.choice 
def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(0,len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x,bs_y,1)

    return bs_slope_reps, bs_intercept_reps

# Generate replicates of slope and intercept using pairs bootstrap
bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(illiteracy,fertility,1000)

# Generate array of x-values for bootstrap lines: x
x = np.array([0,100])
print(x)
# Plot the bootstrap lines
for i in range(100):
    _ = plt.plot(x, 
                 bs_slope_reps[i]*x +bs_intercept_reps[i],
                 linewidth = 0.5, alpha = 0.2, color='red')

# Plot the data
_ = plt.plot(illiteracy,fertility,marker='.',linestyle = 'none')

# Label axes, set the margins, and show the plot
_ = plt.xlabel('illiteracy')
_ = plt.ylabel('fertility')
plt.margins(0.02)
plt.show()
--------------------------------------------------------------------------
# Смотрим,если перемешать данные,то средние значение останутся такими же или нет,потом это можно будет сделать большое кол-во раз и получить распределение
import numpy as np
dem_share_both = np.concatenate((dem_share_PA, dem_share_OH))#соединяем данные 
dem_share_perm = np.random.permutation(dem_share_both)#рандомно перемешиваем данные 
perm_sample_PA = dem_share_perm[:len(dem_share_PA)]# берем одну часть данных(по длине как 'dem_share_PA')
perm_sample_OH = dem_share_perm[len(dem_share_PA):]# берем остальную часть данных (OH)

------
'def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1,data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2
--------



p-value - #Вероятность получения значения вашей тестовой статистики, которое, по крайней мере, является экстремальным по сравнению с тем, что наблюдалось, при допущении, что нулевая гипотеза верна


# Представляем,что michelson_speed_of_light = newcomb_value с сохранением разброса 
newcomb_value = 299860# km/s
michelson_shifted = michelson_speed_of_light - np.mean(michelson_speed_of_light) + newcomb_value


# вычисляем p-value 
# diff_observed - общие данные 
bs_replicates = draw_bs_reps(michelson_shifted,diff_from_newcomb, 10000)
p_value = np.sum(bs_replicates <= diff_observed) / 10000

--------
def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(0,len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x,bs_y,1)

    return bs_slope_reps, bs_intercept_reps
-------

-------
def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data,func)

    return bs_replicates
    ------


import numpy as np
# clickthrough_A, clickthrough_B: arr. of 1s and 0s
def diff_frac(data_A, data_B):
            frac_A = np.sum(data_A) / len(data_A)        
            frac_B = np.sum(data_B) / len(data_B)
    return frac_B - frac_A
diff_frac_obs = diff_frac(clickthrough_A,clickthrough_B)

perm_replicates = np.empty(10000)
for i in range(10000):
    perm_replicates[i] = permutation_replicate(clickthrough_A, clickthrough_B, diff_frac)
         p_value = np.sum(perm_replicates >= diff_frac_obs) / 10000

