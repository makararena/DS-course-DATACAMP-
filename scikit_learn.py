from sklearn import datasets # сам sklearn имеет свои датасеты,которые можно использовать для машинного обучения 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot') # просто используем красивый стиль,который популярен в ML 
iris = datasets.load_iris() #скачиваем датасет iris
type(iris) 'sklearn.utils.Bunch'
print(iris.keys()) # 'data', 'target', 'target_names', 'DESCR', 'feature_names'
type(iris.data), type(iris.target) # np.ndarray
iris.data.shape 
iris.target_names
X = iris.data # выбираем дату,как обучалку 
y = iris.target # выбирает target,как цель 
df = pd.DataFrame(X, columns=iris.feature_names) # создаем dataFrame
print(df.head())

_ = pd.plotting.scatter_matrix(df,
                                 c = y, # цвет меняется по y
                                 figsize = [8, 8], # ширина и длинна матрицы 
                                 s=150, # размер точек, не по диагонали 
                                  marker = 'D') # просто маркер для точек 


from sklearn.neighbors import KNeighborsClassifier # Импортируем Штуку для машиного обучения(график,линии)
knn = KNeighborsClassifier(n_neighbors=6) # заносим штуку для машинного обучения в переменную 
knn.fit(iris['data'], iris['target']) # обучаем по дате и целе
iris['data'].shape #(150, 4)
iris['target'].shape #(150,)
X_new = np.array([[5.6, 2.8, 3.9, 1.1]]) # нужно именно 4,тк в дате у нас тоже 4 колонки 
prediction = knn.predict(X_new) # предиктим по новой целе 
X_new.shape #(,4)
print('Prediction: {}’.format(prediction))





from sklearn.model_selection import train_test_split # импортируем split test(нужен для разделения данных)
X_train, X_test, y_train, y_test = train_test_split(X, # DATA 
                                                    y, # target 
                                                    test_size=0.3, # train - 70% ,test - 30%
                                                    random_state=21, #рандомное число 
                                                    stratify=y) # Если не None, данные разбиваются послойно, используя это как метки класса.
knn = KNeighborsClassifier(n_neighbors=8) # все по старинке,важно указать,что n_neighbours влияет на точность
knn.fit(X_train, y_train) # Тренируем  
y_pred = knn.predict(X_test) # Предиктим 
print(\"Test set predictions:\\n {}\".format(y_pred)
knn.score(X_test, y_test) # Вычисляем точность по тесту 





import numpy as np
from sklearn.linear_model import LinearRegression # Импортируем линейную регрессию
reg = LinearRegression() # создаем линейную регрессию
reg.fit(X_rooms, y) # фитим по дате и цели 
prediction_space = np.linspace(min(X_rooms),max(X_rooms)).reshape(-1, 1) #создаем множество точек по разбросу дынных и решейпим в одну колонку(-1,1 )
plt.scatter(X_rooms, y, color='blue') # создаем синий scatter plot
plt.plot(prediction_space, reg.predict(prediction_space),color='black', linewidth=3) # создаем регрессию X-неог.кол-во точек,y-предиким для каждого X свой y,linewidth - ширина линии 
plt.show()



# Проводим тест с помощью линейной регрессии 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state=42) 
reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)
reg_all.score(X_test, y_test)



from sklearn.model_selection import cross_val_score  
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
cv_results = cross_val_score(reg, X, y, cv=5) # создаем результаты перекрестной проверки,cv - кол-во перекрестных проверок 
print(cv_results)
np.mean(cv_results) # смотрим медиану для перекрестной проверки 



from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state=42)
ridge = Ridge(alpha=0.1, normalize=True) #Гребневая(ридж) регрессия – это регуляризованная версия линейной регрессии.
#Она заставляет алгоритм обучения не только соответствовать данным, но и сохранять веса модели как можно меньшими. Также может увеличит качество модели.
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge.score(X_test, y_test)




from sklearn.linear_model import Lasso
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state=42)
lasso = Lasso(alpha=0.1, normalize=True) # Если Alpha = 0 , то мы получаем линию регрессии,если alpha = 1,то наши коэфициенты значительно ухудшаются(большие коэфициенты)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso.score(X_test, y_test)

from sklearn.linear_model import Lasso
names = boston.drop('MEDV', axis=1).columns 
lasso = Lasso(alpha=0.1) #Наименьшее абсолютное сжатие и регрессия оператора выбора (обычно называется регрессия Лассо)
# – это еще одна регуляризованная версия линейной регрессии
# Старается уменьшить коэфииеты ненужных функций 
lasso_coef = lasso.fit(X, y).coef_ # сохраняем веса 
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation=60)
_ = plt.ylabel('Coefficients')
plt.show()





from sklearn.neighbors import KNeighborsClassifier # Импортируем Штуку для машиного обучения(график,линии)
from sklearn import datasets # сам sklearn имеет свои датасеты,которые можно использовать для машинного обучения 
from sklearn.model_selection import train_test_split # импортируем split test(нужен для разделения данных)
from sklearn.metrics import mean_squared_error # испортируем дисперсию
from sklearn.model_selection import cross_val_score # испортируем перекресную проверку(нужно,чтобы унать точность модели,отбросив при этом разброс)
from sklearn.linear_model import LinearRegression # Импортируем линейную регрессию 
from sklearn.linear_model import Lasso # Импортируем Лассо 
from sklearn.linear_model import Ridgeи # Импортируем Ридж
y = df['party'].values #выделяем определенную колонку и преобразуем в np.array 
X = df.drop('party',axis=1).values # создаем данные(все,кроме определенной колонки) и преобразуем в np.array




---------------------------------------------------------------------------------------------------------------------------------------
# в примере указывается,что,если мы возьмемем 99% правильных имейлов и 1% неправильный,то если наш робот покажет,что все правильные,то точность у нас получится 99%
# чтобы избежать плохих(по-настоящему результатов) мы берем матрицу ответов -> через них мы получаем новые мерики(Presicion,Rekall и тп,которые могут показать точность )
# Precision - не много False Positive 
# Recall - True Positive 
# F1score:2 - их сумма(усложненная)
from sklearn.metrics import classification_report # импортируем репорт,который потом все покажет
from sklearn.metrics import confusion_matrix # импортируем матрицу,которая нужна изначально
knn = KNeighborsClassifier(n_neighbors=8)
X_train, X_test, y_train, y_test = train_test_split(X, y,    test_size=0.4, random_state=42)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(confusion_matrix(y_test, y_pred)) # создаем матрицу 1 - настоящие ответы,2 - то,что мы запредиктили
print(classification_report(y_test, y_pred)) # создаем репорт классификаций 



from sklearn.linear_model import LogisticRegression # испортируем логистическую регрессию
from sklearn.model_selection import train_test_split 
logreg = LogisticRegression() # устанавливаем регрессию
X_train, X_test, y_train, y_test = train_test_split(X, y,    test_size=0.4, random_state=42)
logreg.fit(X_train, y_train) # фитим 
y_pred = logreg.predict(X_test) # предиктим 
    # вообще нужна для того,чтобы наглядно показать,как работает модель(если x = 0,y = 1, то это идеальная модель )
    from sklearn.metrics import roc_curve # импортируем кривую,которая нам покажет логистическую регрессию(для этого вроде и нужна)
    y_pred_prob = logreg.predict_proba(X_test)[:,1] # предиктим X_test с помощью longreg
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob) # вводим три элемента(нужны для кривой) и используем roc_curve
    plt.plot([0, 1], [0, 1], 'k--') # плотим 
    plt.plot(fpr, tpr, label='Logistic Regression')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Logistic Regression ROC Curve')
    plt.show()

# чтобы численно узнать,насколько хороша модель,с помощью ROC
from sklearn.metrics import roc_auc_score # импортируем штуку 
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y,    test_size=0.4, random_state=42)
logreg.fit(X_train, y_train)
y_pred_prob = logreg.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_pred_prob) # код,который выведет потом сам результат 


from sklearn.model_selection import cross_val_score # делаем тоже самое,только с перекрестной проверкой 
cv_scores = cross_val_score(logreg, X, y, cv=5,scoring='roc_auc')  # самое главное - scoring
print(cv_scores)


# тут мы создаем перекрестную проверку тех значений,которые нам нужны(n_neighbors,alpha ......)
from sklearn.model_selection import GridSearchCV # нужно для параметров 
param_grid = {'n_neighbors': np.arange(1, 50)} # создаем dict,который показывает какие числа для соседей мы будем проверять
knn = KNeighborsClassifier() 
knn_cv = GridSearchCV(knn, param_grid, cv=5) # создаем перекрестную проверку 
knn_cv.fit(X, y) # фитим по изначальным данным
knn_cv.best_params_ # узнаем лучшие параметры 
knn_cv.best_score_ # узнаем лучший результат


    # тоже самое,только мы добавили train test 
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    # Create the hyperparameter grid
    c_space = np.logspace(-5, 8, 15)
    param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}
    # Instantiate the logistic regression classifier: logreg
    logreg = LogisticRegression()
    # Create train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.4,random_state = 42)
    # Instantiate the GridSearchCV object: logreg_cv
    logreg_cv = GridSearchCV(logreg,param_grid,cv = 5)
    # Fit it to the training data
    logreg_cv.fit(X_train,y_train)
    # Print the optimal parameters and best score
    print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
    print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))

------------------------------------------------------------------------------------------------------------------
import pandas as pd
df = pd.read_csv('auto.csv')
df_origin = pd.get_dummies(df) #из качественных переменных(Мужчина/Женщина) делает количественные 0/1
df_origin = df_origin.drop('origin_Asia', axis=1) # если у нас более двух переменных,то мы можем сбросить одну без потери смысла 
print(df_origin.head())

df.insulin.replace(0, np.nan, inplace=True) # меняем ненужные нам значения на NaN


from sklearn.pipeline import Pipeline # нужен для последовательного выполнения действий 
from sklearn.preprocessing import Imputer # нужен для изменения значения по колонкам 
imp = Imputer(missing_values='NaN', strategy='mean', axis=0) # меняет все NaN на среднюю по колонке(axis = 0)
logreg = LogisticRegression()
steps = [('imputation', imp),('logistic_regression', logreg)] # выставляем шаги,по которым будет идти машина 
pipeline = Pipeline(steps) # используем шаги
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42) # делаем тест
pipeline.fit(X_train, y_train)# тренируем 
y_pred = pipeline.predict(X_test) # предиктим 
pipeline.score(X_test, y_test) # смотрим результаты 

# модели,которые основываются на дистанции(knn) плохо работают при большом скейле(разбросе данных),поэтому нужно скейлить 
# один из способов - стандартизация (среднее делим на variance)
from sklearn.preprocessing import StandardScaler # импортируем скейлер 
steps = [('scaler', StandardScaler()),('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=21)
knn_scaled = pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
accuracy_score(y_test, y_pred)

# добавляем разделение и вычисление нужного количества n_neighbours
steps = [('scaler', StandardScaler()),(('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
parameters = {knn__n_neighbors: np.arange(1, 50)}
X_train, X_test, y_train, y_test = train_test_split(X, y,    test_size=0.2, random_state=21)
cv = GridSearchCV(pipeline, param_grid=parameters)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
print(cv.best_params_)
print(cv.score(X_test, y_test))
print(classification_report(y_test, y_pred))



x = points[:,0]
y = points[:,1]
plt.scatter(x,y)
plt.show()
