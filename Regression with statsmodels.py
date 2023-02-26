# Принтим корреляцию 
print(swedish_motor_insurance['n_claims'].corr(swedish_motor_insurance['total_payment_sek']))

# Что такое регрессия?
# Если у тебя есть связь между переменной,у которой есть все данные и данные,которые тебе надо создать(другая колонка),
# то идеально подходит регрессия.

# Есть линейная регрессия и логистическая регрессия
# Линейная регрессия - это подгонка прямой линии к данным, а логистическая регрессия - подгонка кривой к данным.
# Линейная регрессия - это алгоритм регрессии для машинного обучения, а логистическая регрессия - это алгоритм классификации для машинного обучения.
# Линейная регрессия предполагает гауссово (или нормальное) распределение зависимой переменной. 
# Логистическая регрессия предполагает биномиальное распределение зависимой переменной

# Плотим линейную регрессию
sns.regplot(x="n_claims",y="total_payment_sek",data=swedish_motor_insurance,ci=None)

# Формула линейной регрессии 
# y = intercept + slope * x

# Использование линейной регрессии 'то,что у нас есть ~ переменная,которая нам интересная(зависимая)'
from statsmodels.formula.api import ols
mdl_mass_vs_species = ols("mass_g ~ species", data=fish).fit()
print(mdl_mass_vs_species.params)


# Так мы делаем,если у нас несколько категориальных переменных -> смотрим intercept -> np.mean()
mdl_mass_vs_species = ols("mass_g ~ species + 0", data=fish).fit()
print(mdl_mass_vs_species.params)

# Предиктим новые данные и присваиваем их нашей таблице,как колонка 'mass_g'
explanatory_data = pd.DataFrame({"length_cm": np.arange(20, 41)})
explanatory_data.assign(mass_g = mdl_mass_vs_length.predict(explanatory_data))



# Как вариант можно запринтить то,как регрессия будет работать с данными,которые мы ей подали на fit 
print(mdl_mass_vs_length.fittedvalues)
# Тоже самое 
explanatory_data = bream["length_cm"]
print(mdl_mass_vs_length.predict(explanatory_data))


# 'Остатки' или (наши данные - данные,которые спрогнозировала наша регрессия)
print(mdl_mass_vs_length.resid)
# Тоже самое 
print(bream["mass_g"] - mdl_mass_vs_length.fittedvalues)

# Общие метрики для нашей модели 
mdl_mass_vs_length.summary()


# Изначально надо понимать,что регрессия принимает в себя нормализованные данные 
# Мы используем : 'кубы','корни','логарифмы' для лучшего распределения рассматриваемой независимой переменной(уменьшить влияние выбросов)



-------------------------------------
# r-squared - метрика,которая говорит,насколько линия регрессии подошла к данным 
# Прогнозируется от 0 до 1(совсем подошла)
# Формула ---- np.corr(df) ** 2 
print(mdl_bream.rsquared)


# RSE - корень из MSE(среднеквадратическая ошибка) - типо насколько в среднем данные в регрессии отличаются от настоящих
# Так можно посчитать 
mse = mdl_bream.mse_resid
rse = np.sqrt(mse)
print("rse: ", rse)


# Считаем кол-во степеней свободы; 2 ,тк у нас регрессия принимает 2 коэфициента   
deg_freedom = len(bream.index) - 2

# RMSE - тоже самое,только оно не учитывает регрессию в количестве степеней свободы(2 коэфициента не отнимает )

# Рисует диаграмму рассеяния остатков
sns.residplot(x="length_cm", y="mass_g", data=bream, lowess=True)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")

# Показывает распределение в квантилях
from statsmodels.api import qqplot
qqplot(data=mdl_bream.resid, fit=True, line="45")

# Тоже самое,что и первое,только остатки под корнем по оси y
model_norm_residuals_bream = mdl_bream.get_influence().resid_studentized_internal
model_norm_residuals_abs_sqrt_bream = np.sqrt(np.abs(model_norm_residuals_bream))
sns.regplot(x=mdl_bream.fittedvalues, y=model_norm_residuals_abs_sqrt_bream, ci=None, lowess=True)
plt.xlabel("Fitted values")
plt.ylabel("Sqrt of abs val of stdized residuals")


# Cмотрим,какие данные сильнее всего влияют на нашу модель(лучше не исползовать hat_diag)
mdl_roach = ols("mass_g ~ length_cm", data=roach).fit()
summary_roach = mdl_roach.get_influence().summary_frame()
roach["leverage"] = summary_roach["hat_diag"]
print(roach.head())

# cook_d пользуется большей популярностью
roach["cooks_dist"] = summary_roach["cooks_d"]

-----------------------------------------------------
# Использование логистической регрессии(график не пересекает значения от 0 до 1)
from statsmodels.formula.api import logit
mdl_churn_vs_recency_logit = logit("has_churned ~ time_since_last_purchase",data=churn).fit()
print(mdl_churn_vs_recency_logit.params)

# Плотим регрессию
sns.regplot(x="time_since_last_purchase",y="has_churned",data=churn,ci=None,logistic=True)

# Мы можем узнать ответы регрессии,просто округлив,тк у нас значения идут от 0 до 1
prediction_data = explanatory_data.assign(has_churned = mdl_recency.predict(explanatory_data))
prediction_data["most_likely_outcome"] = np.round(prediction_data["has_churned"])

# Существует метрика 'odds ratio' -> 'отношение шансов'(вероятность,что что-то произойдет/вероятность того,что не поизойдет)

# Плотим обычную линию по y = 1
plt.axhline(y=1, linestyle="dotted")           

# Создаем confusion matrix(true positive/false negitive....)
actual_response = churn["has_churned"]
predicted_response = np.round(mdl_recency.predict())
outcomes = pd.DataFrame({"actual_response": actual_response,"predicted_response": predicted_response})
print(outcomes.value_counts(sort=False))

# Тут мы просто переводим в матричный вид(не совтую,тк не поймешь где TP/TN либо сиди,читай,как это сделать)
conf_matrix = mdl_recency.pred_table()

# Импортируем график,который это все представляет в красивом виде
from statsmodels.graphics.mosaicplot import mosaic
mosaic(conf_matrix)

# Метрика 'accuracy' вычисляеться так:
TN = conf_matrix[0,0]
TP = conf_matrix[1,1]
FN = conf_matrix[1,0]
FP = conf_matrix[0,1]
acc = (TN + TP) / (TN + TP + FN + FP)
print(acc)

# Метрика 'sensivity' вычисляеться так:
sens = TP / (FN + TP)
print(sens)

# Метрика 'Specifity' вычисляеться так:
spec = TN / (TN + FP)
print(spec)
