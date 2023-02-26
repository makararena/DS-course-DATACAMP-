is_black_or_brown = dogs["color"].isin(["Black", "Brown"])
# просто забыл про это код

median(),.mode().min(),.max().var(),.std().sum().quantile()
# считалки 

dogs["weight_kg"].agg(pct30)
# agg-шка

.cummax().cummin().cumprod()
# еще считалки 

vet_visits.drop_duplicates(subset=["name", "breed"])
# сбрасываем дубликаты 

value_counts(normslize,sort)
#cчитаем 

dogs.groupby("color")["weight_kg"].mean()
# группирока 

dogs.pivot_table(values="weight_kg", index="color", columns="breed",fill_value=0, margins=True)
# создание своей сводной таблицы

dogs_ind = dogs.set_index("name")
# добавляем индекс 

dogs_ind.reset_index(drop = True )
# убираем индекс и удаляем значения 

dogs_ind3.loc[[("Labrador", "Brown"), ("Chihuahua", "Tan")]]
# использование loc с двойными индексами 

dogs_ind3.sort_index(level=["color", "breed"], ascending=[True, False])  
# сортируем

dogs_srt.loc[("Labrador", "Brown"):("Schnauzer", "Grey"), "name":"height_cm"]
# loc двойного индекса 

private_employee = employee[['employee_id', 'salary']]
# чтобы сохранить именно таблицу [[]]

IQR = stats.iqr(pH)
#высчитывание iqr

missing_total_by_column = wine.isna().sum()
# находим сумму недостающих значений

print(employee.index)
#просто смотрим индексы 

print(df.drop(columns="repo"))
#просто удаляем колонку

! ls 'file'
#помогает посмотреть,импортирован ли такой файл
