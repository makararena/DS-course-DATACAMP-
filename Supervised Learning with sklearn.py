# Типы обучения с учителем 
# 1.Классификация(бинарная и другие)
# 2.Регрессия

# Feature = predictor variable = independent variable
# Target variable = dependent variable = response variable

# Перед тем,как использовать модель 
# 1. No missing values
# 2. Data in numeric format 
# 3. Data stored in padas or numpy 

# Importing most common model
from sklearn.module import Model
model = Model()
model.fit(X, y)
predictions = model.predict(X_new)
print(predictions)

# Using KNN 
# Working on по принципу of nearest neighbors 
# Подробнее лучше посмотри картинки(чем большее количество соседей,тем большую область около точки ищет,тем сложнее модель)
# Может быть проблема в переобучении и недообучении 
from sklearn.neighbors import KNeighborsClassifier
X = churn_df[["total_day_charge", "total_eve_charge"]].values
y = churn_df["churn"].values
print(X.shape, y.shape)
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X, y)

# Использование train_test_split и score(в принципе все понятно)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=21, stratify=y) # stratify отвечает за то,чтобы возвращало 
# определенный процент выборок класса,который нам нужен(тут это y),то есть перемешивает данные,но сохраняет внешний вид 
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))

# Плотим разнае количество neighbors,чтобы узнать то количество,которое нам лучше всего подходит 
train_accuracies = {}
test_accuracies = {}
neighbors = np.arange(1, 26)
for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

plt.figure(figsize=(8, 6))
plt.title("KNN: Varying Number of Neighbors")
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show()


# Плотим самую простую линейную регрессию с нашими данными 
# Самая простая регрессия работает по OLS(минимизирует RSS(функцию потерь(ищет самое маленькое расстояние от прямой до всех точек)))
# Можно использовать сразу несколько переменных,если они сильно влияют 
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_bmi, y)
predictions = reg.predict(X_bmi)
plt.scatter(X_bmi, y)
plt.plot(X_bmi, predictions)
plt.ylabel("Blood Glucose (mg/dl)")
plt.xlabel("Body Mass Index")
plt.show()


# Скоринг в регрессии отвечает за R-squared(R**2) - если линия отвечает данным и отход точек от линии небольшой,то scoring стремится к 1,если
# наоборот,то и скоринг буде меньше 
reg_all.score(X_test, y_test)

# Показано использование среднеквадратической ошибки
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred, squared=False) # если squared - True,то возвращает MSE,если false,то RMSE

# Использование перекрестной проверки + KFold(создаем кол-во cv + перемешиваем данные)
from sklearn.model_selection import cross_val_score, KFold
kf = KFold(n_splits=6, shuffle=True, random_state=42)
reg = LinearRegression()
cv_results = cross_val_score(reg, X, y, cv=kf)

# Считаем квантиль(95-й процентиль)
print(np.quantile(cv_results, [0.025, 0.975]))

# Ridge работает немного по-другому(убирает выбросы(сильно штрафует модель за них))
from sklearn.linear_model import Ridge
scores = []
for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        ridge = Ridge(alpha=alpha)    # alpha - примерно как n_neighbors,если alpha = 0,то это OLS,дальше начинаеься penalizing 
        ridge.fit(X_train, y_train)    
        y_pred = ridge.predict(X_test)    
        scores.append(ridge.score(X_test, y_test))
print(scores)

# Лассо же наоборот убирает значения,которые не сильно влияют(очень удобна,чтобы наглядно посмотреть,какие именно колонки влияют больше всего)
from sklearn.linear_model import Lasso
scores = []
for alpha in [0.01, 1.0, 10.0, 20.0, 50.0]:  
    lasso = Lasso(alpha=alpha)  
    lasso.fit(X_train, y_train)  
    lasso_pred = lasso.predict(X_test)  
    scores.append(lasso.score(X_test, y_test))
print(scores)

# Вот так показываем самые важные колонки(которые сильнее всего влияют на результат)
from sklearn.linear_model import Lasso
X = diabetes_df.drop("glucose", axis=1).values
y = diabetes_df["glucose"].value
names = diabetes_df.drop("glucose", axis=1).columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_ # берем коэфициенты 
plt.bar(names, lasso_coef)
plt.xticks(rotation=45)plt.show()


# Использование confusion matrix и classification report(подробно о метриках и confusion matrix написал в тетради)
# а classification report просто сборка из всех метрик 
from sklearn.metrics import classification_report, confusion_matrix
knn = KNeighborsClassifier(n_neighbors=7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=42)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Использование Логистической Регресии(рабоает с классификацией) значение от 0 до 1,порог 0.5 по дефолту
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
# Предиктим вероятность того или иного ответа 
y_pred_probs = logreg.predict_proba(X_test)[:, 1]
print(y_pred_probs[0])


# roc curve очень сильно помогает,тк мы меняем это значение(0.5) и впоследствии у нас регрессия подстраиваеться под данные и работает просто лучше
# fpr - false positive rate 
# tpr - true positive rate
# thresholds - понятно 
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()

# roc_auc_score считает площадь под threshold,чем ближе к 1,тем лучше(по дефолту 0.5)
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred_probs))

# GridSearchCV удобна тем,что мы берем сразу несколько alpha и других параметров и выбираем лучший,что урезает код и просто легче,чем рисовать графики
# Главное не брать большое количество,тк оно плохо масштабируется 
from sklearn.model_selection import GridSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {"alpha": np.linspace(0.0001, 1, 10),"solver": ["sag", "lsqr"]}
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid, cv=kf)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)

# тут уже используем рандомизированную SeachCV,работает быстрее 
from sklearn.model_selection import RandomizedSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {'alpha': np.linspace(0.0001, 1, 10),"solver": ['sag', 'lsqr']}
ridge = Ridge()
ridge_cv = RandomizedSearchCV(ridge, param_grid, cv=kf, n_iter=2) # n_iter перемножается с n_splits,тем самым получаем 10 рандомных значений,
# из которых уже смотрим лучшее
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)



# используем pd.get_dummies,чтобы нашу категориальную колонку представить в двоичном формате(лучше показано на фото( 0 0 0 0 0 1 0 ))
import pandas as pd
music_df = pd.read_csv('music.csv')
music_dummies = pd.get_dummies(music_df["genre"], drop_first=True) # используем drop_first,тк нам одна колонка не нужна,тк если везде нули,то точно 
# эта колонка 
print(music_dummies.head())


# тут сам прикол в neg_mean_squared_error(мы ее используем,тк с категориальными переменными так рабоатеться лучше) и потом все возвращаем и смотрим 
# дисперсию
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
X = music_dummies.drop("popularity", axis=1).values
y = music_dummies["popularity"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
linreg = LinearRegression()
linreg_cv = cross_val_score(linreg, X_train, y_train, cv=kf,scoring="neg_mean_squared_error")
print(np.sqrt(-linreg_cv))


# SimpleImputer нужен для того,чтобы заполнять NaN значениями(в этом примере самыми частыми) категориальные переменные(столбцы)
from sklearn.impute import SimpleImputer
X_cat = music_df["genre"].values.reshape(-1, 1)
X_num = music_df.drop(["genre", "popularity"], axis=1).values
y = music_df["popularity"].values
X_train_cat, X_test_cat, y_train, y_test = train_test_split(X_cat, y, test_size=0.2,random_state=12)
X_train_num, X_test_num, y_train, y_test = train_test_split(X_num, y, test_size=0.2,random_state=12)
imp_cat = SimpleImputer(strategy="most_frequent")
X_train_cat = imp_cat.fit_transform(X_train_cat)
X_test_cat = imp_cat.transform(X_test_cat)

# А так мы уже работам с не категориальными данными с помощью Imputer
imp_num = SimpleImputer()
X_train_num = imp_num.fit_transform(X_train_num)
X_test_num = imp_num.transform(X_test_num)
X_train = np.append(X_train_num, X_train_cat, axis=1)
X_test = np.append(X_test_num, X_test_cat, axis=1)

# Используем Пайплайн 
from sklearn.pipeline import Pipeline
music_df = music_df.dropna(subset=["genre", "popularity", "loudness", "liveness", "tempo"])
music_df["genre"] = np.where(music_df["genre"] == "Rock", 1, 0) # Столбец 'genre',если есть Rock заполяем 1,если нету,то 0(оставляем только Rock)
X = music_df.drop("genre", axis=1).values
y = music_df["genre"].values
# Используем Пайплайн 2.0
steps = [("imputation", SimpleImputer()),("logistic_regression", LogisticRegression())]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)

# Используем StandartScaler - дисперсия - 1, и все данные находятся около 0 -- стандартизируем данные 
from sklearn.preprocessing import StandardScaler
X = music_df.drop("genre", axis=1).values
y = music_df["genre"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(np.mean(X), np.std(X))
print(np.mean(X_train_scaled), np.std(X_train_scaled))
steps = [('scaler', StandardScaler()),('knn', KNeighborsClassifier(n_neighbors=6))]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=21)
knn_scaled = pipeline.fit(X_train, y_train)
y_pred = knn_scaled.predict(X_test)
print(knn_scaled.score(X_test, y_test))

# Перемешиваем все вместе (GridSearchCV,STandartScaler и Pipeline)
from sklearn.model_selection import GridSearchCV
steps = [('scaler', StandardScaler()),('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
parameters = {"knn__n_neighbors": np.arange(1, 50)}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=21)
cv = GridSearchCV(pipeline, param_grid=parameters)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
print(cv.best_score_)   # Смотрим лучший результат 
print(cv.best_params_)  # Смотрим лучший параметр 


# Разные модели предназначения для разных вещей,поэтому важно правильно все выбрать исходя из того,что ты знаешь,сколько есть времени и какие 
# результаты тебе нужны 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
X = music.drop("genre", axis=1).values
y = music["genre"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {"Logistic Regression": LogisticRegression(), "KNN": KNeighborsClassifier(), "Decision Tree": DecisionTreeClassifier()}
results = []
for model in models.values():
         kf = KFold(n_splits=6, random_state=42, shuffle=True)     
         cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)     
         results.append(cv_results)
         plt.boxplot(results, labels=models.keys())
         plt.show()
         
for name, model in models.items():
      model.fit(X_train_scaled, y_train)  
      test_score = model.score(X_test_scaled, y_test)
      print("{} Test Set Accuracy: {}".format(name, test_score))


# Работает почти также как LinearRegression(по - другому предиктит)
svm = LinearSVC()
