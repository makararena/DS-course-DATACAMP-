# читаем данные hdf и сохраняем в переменную nsfg 
nsfg = pd.read_hdf('nsfg.hdf5', 'nsfg')

# смотрим,какие колонки есть у нас 
nsfg.columns

# сначала считаем количество значений какой-то колонки по индексу и потом сортируем по индексу 
pounds = nsfg['birthwgt_lb1']
pounds.value_counts().sort_index()

#считаем min,max,std и тп 
pounds.describe()

# Меняем значение 98,99 на Nan и сохраняем это все в исходной таблице
ounces.replace([98, 99], np.nan, inplace=True)

# Убираем все Nan
birth_weight.dropna()

#Мжно вычислить среднюю таблицы там,где True/False,также сумму и медиану 

birth_weight [A & B]    # Логическое "и" 
birth_weight[A | B]     # Логическое "или"

#Функция вероятности, также называемая числовой плотностью 
# просто считаем какая есть разновидность у данных(1.0 - 578,2.0 - 78457),если мы используем Normalize = True,то мы найдем именно процентность данных(1.0 - 0.33)
pmf_educ = Pmf(educ, normalize=False/True)


# CDF показывает вероятность того,что ты получишь значение <=x
df = Cdf(gss['age'])

# код показывает,сколько процентов значений будет <= x(0.66)
q = 51
p = df(q)
print(p)

# код показывает, до какого значения может дойти,если CDF будет(0.25),то есть 25%
p = 0.25
q = df.inverse(p)
print(q)

#создаем среднее распределение(постоянное значение,не меняется),которое связанно с cdf (cumsum)
# будет выглядеть в виде волны 
from scipy.stats import norm
xs = np.linspace(-3, 3)#range
ys = norm(0, 1).cdf(xs)
plt.plot(xs, ys, color='gray')
plt.show()

# cоздаем кривую гауса,в виде колокола,которая нам показывает распределение,если мы будем использовать PMF
# универсальное средство выражения количественного распределения в обществе массовых социальных свойств, признаков, черт, явлений, процессов и т.д
xs = np.linspace(-3, 3)
ys = norm(0,1).pdf(xs)
plt.plot(xs, ys, color='gray')


#Cмотрим KDE 
#Ядерная оценка плотности является задачей сглаживания данных, когда делается заключение о совокупности, основываясь на конечных выборках данных. 
sns.kdeplot(sample)


#Use CDF for exploration
#Use PMF if there are a small number of unique values
#Use KDE if there are a lot of values


height_jitter = height + np.random.normal(0, 2, size=len(brfss))# добавляем дрожание высоте 
weight_jitter = weight + np.random.normal(0, 2, size=len(brfss)) # добавляем дрожение весу 
plt.plot(height_jitter, weight_jitter, 'o', markersize=1, alpha=0.01) # плотим это все 'o'-кружочки какие должны быть,markersize - размер кружочков,alpha-прозрачность 
plt.axis([140, 200, 0, 160]) # зумим картинку(сначала по x,потом по y)
plt.yscale('log')# скелим по y на определенное значение(тут у нас log)
plt.show()



# Вычисляем корреляцию  между определенными столбиками
columns = ['HTM4', 'WTKG3', 'AGE']
subset = brfss[columns]
subset.corr()

# высчитываем регрессионные данные,такая штука используется,если есть линейные отношения между переменными 
from scipy.stats import linregress
res = linregress(xs, ys)
#Главное-slope-наклон-коррелляция

# добавляем линию регресиии 
fx = np.array([xs.min(), xs.max()])
fy = res.intercept + res.slope * fx
plt.plot(fx, fy, '-')


# важно понимать одно: представим,что у нас есть две линии регресии:x и y 
# X может зависеть от y не так сильно,как y от x для этого мы исползуем одну функцию,которая поможет нам установить,какую именно зависимость мы хоти найти 
import statsmodels.formula.api as smf
results = smf.ols('INCOME2 ~ _VEGESU1', data=brfss).fit() #устанавливаем зависимость и фитим ее 
results.params #находим параметры(коррелцию) или же наклон Vegesu1-cнизу,income-слева

# здесь мы добвляем новый параметр 'age',который также считается в зависимости от realinc 
results = smf.ols('realinc ~ educ + age', data=gss).fit()
results.params

# здесь мы добавили новый параметр 'age2',который покажет совсем другую картину,отличающуюся от age,тк это уже у нас нелинейная регрессия(по смыслу)
gss['age2'] = gss['age']**2
model = smf.ols('realinc ~ educ + age + age2', data=gss)
results = model.fit()
results.params



# здесь уже мы создаем модель предсказаний на основе данных из таблиц 

results = smf.ols('realinc ~ educ + educ2 + age + age2', data=gss).fit()

# Make the DataFrame
df = pd.DataFrame()
df['educ'] = np.linspace(0,20)
df['age'] = 30
df['educ2'] = df['educ']**2
df['age2'] = df['age']**2
grouped = gss.groupby('educ')#группируем по 'educ'
mean_income_by_educ = grouped['realinc'].mean()#высчитываем среднее для каждых доходов
plt.plot(mean_income_by_educ,'o',alpha = 0.5) # показываем средние результаты 

# Plot the predictions
pred = results.predict(df) #создаем предсказания на основе данных из 'results'
plt.plot(df['educ'], pred, label='Age 30') # добавляем результаты на график 

# Label axes
plt.xlabel('Education (years)')
plt.ylabel('Income (1986 $)')
plt.legend()
plt.show()



# Логистическая регрессия нужна для сравнения результатов между категорияями М/Ж , Да/Нет
# Recode grass
gss['grass'].replace(2, 0, inplace=True)# просто заменяем 0 на 2,чтобы у нас ответы были только либо 1 либо 2

# Run logistic regression
results = smf.logit('grass ~ age + age2 + educ + educ2 + C(sex)', data=gss).fit()#создаем формулу,добавляя колону С(categotial),с полом 
results.params # смотрим резултаты 

df = pd.DataFrame()
df['age'] = np.linspace(18, 89)#добавляем range от 18 до 89
df['age2'] = df['age']**2 #считаем среднеквадратичное 
df['educ'] = 12 #добавляем время едукации 
df['educ2'] = df['educ']**2 #добавляем среднеквадратичное 

# Generate predictions for men and women
df['sex'] = 1
pred1 = results.predict(df)# predict results,based on mans 
df['sex'] = 2
pred2 = results.predict(df)#predict results,based on women 
plt.clf()

grouped = gss.groupby('age')#группировка по возрасту 
favor_by_age = grouped['grass'].mean()#высчитываем среднее для каждого возраста 

plt.plot(favor_by_age, 'o', alpha=0.5) # добавляем к графику точки 
plt.plot(df['age'], pred1, label='Male')# плотим мужчин
plt.plot(df['age'],pred2,label = 'Female')# плотим женщин 

plt.xlabel('Age')
plt.ylabel('Probability of favoring legalization')
plt.legend()
plt.show()
