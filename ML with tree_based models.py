# Import DecisionTreeClassifier
# Это такая штука,которая использует if/else для принятия решение(в какую секту будут поступать данные)
# Также неплохо заметить,что эта штука работает на scatter plot квадратами(прямоугольниками),вместо прямой,как та же регрессия
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
dt = DecisionTreeClassifier(max_depth=2, random_state=1) # max_depth - максимальная глубина дерева
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
accuracy_score(y_test, y_pred)

Root:no parent node,question giving rise to two children nodes.
Internal node:one parent node, question giving rise to two children nodes.
Leaf:one parent node,no children nodes-->prediction.
Branch : палка,которая соединяет эти листья 

# Все эти штуки работают до того момента,пока мы не укажем либо глубину самого дерева,либо глубины информации,входящей в листь
# Все работают по критериям(criterion)(их два gini,entropy)

dt = DecisionTreeClassifier(criterion='gini', random_state=1)


from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE # среднеквадратическая ошибка - дисперсия 
X_train, X_test, y_train, y_test= train_test_split(X, y,test_size=0.2,random_state=3)
dt = DecisionTreeRegressor(max_depth=4,min_samples_leaf=0.1,random_state=3)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
mse_dt =  MSE(y_test, y_pred)
rmse_dt = mse_dt**(1/2) # вычитаем корень и у нас получается RMSE 
print(rmse_dt) 




# Модель не должна быть слишком сложной и не должна быть слишком простой

#f = bias(смещение)**2 + variance(дисперсия) + error 

# Как понять,есть ли у нас с чем-то из этого проблемы 
# 1. Сплитим данные на тренировочный и тестовый набор 
# 2. Фитим тренировочный набор
# 3. Оцениваем ошибку на тестовом наборе 
# 4. Смотрим,какие ошибки чему соответствуют
# * тестовый набор не должен будет задействован до того момента,пока мы не уверены в модели,поэтому мы будем использовать перекрестную проверку

# Если у f высокая variance,CV MSE of f > train set MSE -----> переобучение данных
# Способы решения: decreasing model complexity,increase min_samples_leaf,decrease max_depth,gather more data

# Если у f высокий bias : CV MSE of f = train set MSE f = desired error -----> недообучение данных
# Способы решения: increase model complexity,increase max_depth,decrease min_samples_leaf,gather more relevant features

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score
SEED = 123
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=SEED)
dt = DecisionTreeRegressor(max_depth=4,min_samples_leaf=0.14,random_state=SEED)
# Важно обратить внимание на cross_val_score
MSE_CV = - cross_val_score(dt, X_train, y_train, cv= 10,scoring='neg_mean_squared_error',n_jobs = -1)
dt.fit(X_train, y_train)
y_predict_train = dt.predict(X_train)
y_predict_test = dt.predict(X_test)
# CV MSE  
print('CV MSE: {:.2f}'.format(MSE_CV.mean()))
# Training set MSE
print('Train MSE: {:.2f}'.format(MSE(y_train, y_predict_train)))
# Test set MSE
print('Test MSE: {:.2f}'.format(MSE(y_test, y_predict_test)))
---
# Import functions to compute accuracy and split data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Import models, including VotingClassifier meta-model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import VotingClassifier # Дико крутая штука,которая позволяет соединить все модели в одну модел и получить самый точный результат 
# оно все модели фитим по данным(конечно в разных моделях будут разные ответы и наш VotingClassifier берет большинство ответов как за свой ответ)
# Set seed for reproducibility
SEED = 1
# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size= 0.3,random_state= SEED)
# Instantiate individual classifier
lr = LogisticRegression(random_state=SEED)
knn = KNN()
dt = DecisionTreeClassifier(random_state=SEED)
# Define a list called classifier that contains the tuples (classifier_name, classifier)
 classifiers = [('Logistic Regression', lr),
                 ('K Nearest Neighbours', knn),               
                 ('Classification Tree', dt)]
# Iterate over the defined list of tuples containing the classifiers
for clf_name, clf in classifiers:
    #fit clf to the training set    
    clf.fit(X_train, y_train)
    # Predict the labels of the test set    
    y_pred = clf.predict(X_test)
    # Evaluate the accuracy of clf on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy_score(y_test, y_pred)))
# Instantiate a VotingClassifier 'vc'
vc = VotingClassifier(estimators=classifiers)
 # Fit 'vc' to the traing set and predict test set labels
 vc.fit(X_train, y_train)   
 y_pred = vc.predict(X_test)
 # Evaluate the test-set accuracy of 'vc'
 print('Voting Classifier: {.3f}'.format(accuracy_score(y_test, y_pred)))



 # Bagging Classifier - новая модель(если до этого мы собирали как можно больше моделей и тренировали их по данным
 # то сейчас мы берем модель и тренируем по одному датасету,который разбиваем на много маленьких и перемешиваем их сежду собой 
 # например,если у нас был сет ABC,то мы можем тренировать на ABB,BBC,ССС,AAA и тп)
 from sklearn.ensemble import BaggingClassifier 
 from sklearn.tree import DecisionTreeClassifier
 from sklearn.metrics import accuracy_score
 from sklearn.model_selection import train_test_split
 SEED = 1 
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y,random_state=SEED)
 dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16, random_state=SEED)
 bc = BaggingClassifier(base_estimator=dt, n_estimators=300, n_jobs=-1) # n_etimators - количество этих разбиений
 bc.fit(X_train, y_train)
 y_pred = bc.predict(X_test)
 accuracy = accuracy_score(y_test, y_pred)
 print('Accuracy of Bagging Classifier: {:.3f}'.format(accuracy))


# При OOB = True мы берем разные семпллы и в них делим на Test set и Train Set 
# После этого в каждом семпле вычисляем точность и это будет OOB1 и после всех разбиений 
# Мы это все складываем и делим на количество OOB по итогу у нас получается неплохая метрика точности 
bc = BaggingClassifier(base_estimator=dt, n_estimators=300,oob_score=True, n_jobs=-1)
bc.fit(X_train, y_train)
y_pred = bc.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
oob_accuracy = bc.oob_score_ # Находим точность и записываем ее в переменную
print('Test set accuracy: {:.3f}'.format(test_accuracy))
print('OOB accuracy: {:.3f}'.format(oob_accuracy))


--------------------------------------------------------------------------------------------
# Рандомный Лес - тоже самое,что и Bagging(сделан на его основе),но добавили рандомизацию 
# У нас есть огромное кол-во разбиений и там мы смотрим итог по этому разбриению и выводим потом итоги по другим разбиениям и общим итогом будет ответ,включающий в себя большее число ответов 
rf = RandomForestRegressor(n_estimators=400,min_samples_leaf=0.12,random_state=SEED) 
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rmse_test = MSE(y_test, y_pred)**(1/2)
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))\
# Смотрим,какие колонки в таблице,на которой мы строим модель - самые важные 
import pandas as pd
import matplotlib.pyplot as plt
importances_rf = pd.Series(rf.feature_importances_, index = X.columns)                               
sorted_importances_rf = importances_rf.sort_values()   
sorted_importances_rf.plot(kind='barh', color='lightgreen'); plt.show()



# AdaBoostClassifier делает так,чтобы наша модель училась сама с себя n-е количество раз + измняет веса 
# roc_auc_score - новая метрика, но для этого нам нужен позитивный класс predict proba 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score 
dt = DecisionTreeClassifier(max_depth=1, random_state=SEED)
# Instantiate an AdaBoost classifier 'adab_clf'
adb_clf = AdaBoostClassifier(base_estimator=dt, n_estimators=100)
# Fit 'adb_clf' to the training set
adb_clf.fit(X_train, y_train)
# Predict the test set probabilities of positive class
y_pred_proba = adb_clf.predict_proba(X_test)[:,1]
# Evaluate test-set roc_auc_score
adb_clf_roc_auc_score = roc_auc_score(y_test, y_pred_proba)
print('ROC AUC score: {:.2f}'.format(adb_clf_roc_auc_score))

# GradientBoostingRegressor работает по аналогии с AdaBoostClassifier,только не использует веса семплов 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as MSE
gbt = GradientBoostingRegressor(n_estimators=300, max_depth=1, random_state=SEED,
                                subsample=0.8) # Уменьшение приводит к уменьшению диспресии и увеличении стандартной ошибки 
gbt.fit(X_train, y_train)
y_pred = gbt.predict(X_test)
rmse_test = MSE(y_test, y_pred)**(1/2) # вычисляем среднеквадратическое отклонение
print('Test set RMSE: {:.2f}'.format(rmse_test))
------------------------------------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
SEED = 1
dt = DecisionTreeClassifier(random_state=SEED)
print(dt.get_params()) # Получаем все параметры,которые мы можем изменить в модели
# GridSeachCV - штука,которая работает как CrossValidation,только дополнительно создает матрицу с параметрами и прогоняет эти параметры по всем
# возможным сценариям
from sklearn.model_selection import GridSearchCV
params_dt = {'max_depth': [3, 4,5, 6],
            'min_samples_leaf': [0.04, 0.06, 0.08],
            'max_features': [0.2, 0.4,0.6, 0.8]            
            }
grid_dt = GridSearchCV(estimator=dt,
                        param_grid=params_dt,                       
                        scoring='accuracy',                                              
                        cv=10,                       
                        n_jobs=-1)
grid_dt.fit(X_train, y_train)
# Смотрим лучшие параметры,которые были использованы 
best_hyperparams = grid_dt.best_params_
print('Best hyerparameters:\n', best_hyperparams)
# Смотрим лучшую точность по CV 
best_CV_score = grid_dt.best_score_
print('Best CV accuracy'.format(best_CV_score))
# Создаем лучшую модель,которую мы получили 
best_model = grid_dt.best_estimator_
# Смотрим точность этой модели 
test_acc = best_model.score(X_test,y_test)
print("Test set accuracy of best model: {:.3f}".format(test_acc)) # {:.3f} - мы округляем,теперь мы получаем только 3 знака поссле запятой  
