import seaborn as sns
sns.distplot(df['alcohol']
                kde=False,# убираем ли KDE
                bins=10 # корзинки 
                hist=False, #убираем ли саму гистограмму 
                rug=True # добавляем или убираем маргинальные распределения 
                kde_kws={'shade':True} # добавляем или убираем тень для KDE на всем графике  
#создает график,подобный histogram + by default calculates KDE(Это непараметрический способ оценки плотности случайной величины.->
#->Ядерная оценка плотности является задачей сглаживания данных, когда делается заключение о совокупности,->
# ->основываясь на конечных выборках данных. )

sns.regplot(data=df, x='temp',
            y='total_rentals',
            marker='+'#добавление маркеров 
            order=2 # добавление полиномиальной регрессии
            x_jitter=.1 #добавление категориальных значений 
            x_estimator=np.mean # высчитывает среднее по определенным значениям
            x_bins=4)#добавление корзинок 
# что-то по типу scatterplot,но более крутая версия 

sns.lmplot(x="quality",
           y="alcohol",           
           data=df,           
           col="type" #добавление переменной для колоночек 
           hue="type" #добавление цвета 
           fit_reg=False) #убираем линию 
# lmplot тоже самое,что и regplot,только более крутая версия и поддерживает больше функций,также считает маргинальное распределение 




for style in ['white','dark','whitegrid','darkgrid','ticks']:
        sns.set_style(style)    
        sns.distplot(df['Tuition'])    
plt.show()
# показ сразу нескольких стилей графиков 

sns.despine(left=True)
# убираем левый 'ободок'

sns.set(color_codes=True)
sns.distplot(df['Tuition'], color='g')
#добавление своих цветов 

for p in sns.palettes.SEABORN_PALETTES:
        sns.set_palette(p)
        sns.distplot(df['Tuition'])
plt.show()
#использование сразу нескольких палитр(всех самых regular

 for p in sns.palettes.SEABORN_PALETTES:
         sns.set_palette(p) #function displays a palette  
         sns.palplot(sns.color_palette())   #returns the current palette
plt.show())
#показ сразу нескольких палитр 


fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2, sharey=True, figsize=(7,4))#создание сразу двух графиков,sharey-приравние графиков между собой,figsize-изменяет размер рисунка 

sns.distplot(df['Tuition'], ax=ax0)#показ на первый график 
sns.distplot(df.query('State == "MN"')['Tuition'], ax=ax1)#показ на второй график 

ax1.set(xlabel="Tuition (MN)", xlim=(0, 70000))#выставляем лимит и название на первый график 
ax1.axvline(x=20000, label='My Budget', linestyle='--')#добавляем линию на второй график
ax1.legend()
#Большое количество изменений делается,если мы работаем с осями 

----------------------------------------------------------------------------------------------------------------------------------------------
#Разные типы графиков отвечают за разные значения 
# 1.Show each observation(stripplot(jitter = True -> убирает среднюю линию ),swarmplot)
# 2.Abstract representations(boxplot,violinplot,lvplot)
# 3.Statistical estimates(barplot,pointplot,countplot)

----------------------------------------------------------------------------------------------------------------------------------------
pd.crosstab(df["mnth"], df["weekday"],values=df["total_rentals"],aggfunc='mean').round(0)
# создание матрицы из года и недели и добавление в значение ["total_rentals"] и высчитавание среднего и потом уже округление 
pd.crosstab()
#позволяет создавать матрицу 

sns.heatmap(df_crosstab,
            annot=True,#добавление ячеек с числами 
            fmt="d"#добавляет сразу несколько легенд справа,хз,зачем надо 
            cmap="YlGnBu",#добавление палитры цветов
            cbar=True,#добавление легенды справа 
            center=df_crosstab.loc[9, 6]#добавление центра 
            linewidths=.5)#добавляем промежутки меду ячейками 


sns.heatmap(df.corr())
#высчитывание кореляции в тепловой карте


# FacetGrid() -> factorplot() -> lmplot() все это решетчатые графики 
sns.factorplot(x="Tuition",
               data=df,               
               col="HIGHDEG", 
               kind='box')
# пример factorplot

g = sns.FacetGrid(df, col="HIGHDEG")
g.map(plt.scatter, 'Tuition', 'SAT_AVG_ALL')
#пример FacetGrid(),всегда двухступенчатое 
----
g = sns.PairGrid(df, vars=["Fair_Mrkt_Rent","Median_Income"])
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter)
# использование PairGrid() работает похожим образом,как FacedGrid,но значения мы меняем по диагонали 

ns.pairplot(df.query('BEDRMS < 3'),
             vars=["Fair_Mrkt_Rent","Median_Income", "UTILITY"],
             hue='BEDRMS', palette='husl',             
             plot_kws={'alpha': 0.5})
# использование pairplot() более легкая версия PairGrid 

-----
g = sns.JointGrid(data=df, x="Tuition",
                  y="ADM_RATE_ALL")
 g = g.plot_joint(sns.kdeplot)
 g = g.plot_marginals(sns.kdeplot, shade=True)
 g = g.annotate(stats.pearsonr)
# использование JointGrid() и добавление kdeplot

sns.jointplot(data=df, x="Tuition",
              y="ADM_RATE_ALL", kind='hex')
# использование jointplot() и построение сотового графика
