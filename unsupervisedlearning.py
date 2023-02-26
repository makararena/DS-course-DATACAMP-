from sklearn.cluster import KMeans # испортируем модель,которая может разделить все наши данные по определенным группам
model = KMeans(n_clusters=3) # создаем модель,n_clusters - количество групп,на которые мы хочем разделить данные 
model.fit(samples) # фитим 
labels = model.predict(samples) # предиктим 
print(labels) # получаем результат 
new_labels = model.predict(new_samples) # предиктим по новым данным
print(new_labels)


# чтобы изначально понимать,сколько кластеров нам нужно,мы можем создать scatter plot 
import matplotlib.pyplot as plt
xs = samples[:,0]
ys = samples[:,2]
plt.scatter(xs, ys, c=labels)
plt.show()

# Важная штука,которая показывает нсколько точно работает модель
import pandas as pd
df = pd.DataFrame({'labels': labels, 'species': species}) # создаем датафрейм с двумя колонками( labels - то,что мы запредиктили(0,1,2),species - наши ответы)
ct = pd.crosstab(df['labels'], df['species']) # делаем перекрестную таблицу,которая покажет сколько 0,1,2 соответствует нужным значениям
print(ct)



from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(samples)
print(model.inertia_) # показывает,наколько точно у нас работает кластеризация(чем меньше результат,тем лучше,но нам лучше всего использовать
# именно то количество кластеров,где наблюдается самый высокий спад)


# код,который помогает посмотреть,какое кол-во кластеров нам нужно
ks = range(1, 6)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters = k)
    # Fit model to samples
    model.fit(samples)
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    # Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


from sklearn.preprocessing import StandardScaler # импортируем стандартный скейлер(среднее становится нулем,а дисперсия становится единицей)
# помогает сделать так,чтобы модель лучше работала 
from sklearn.cluster import KMeans
scaler = StandardScaler()
kmeans = KMeans(n_clusters=3)
from sklearn.pipeline import make_pipeline # импортируем пайплайн(make_pipeline)
pipeline = make_pipeline(scaler, kmeans) 
pipeline.fit(samples)

#MaxAbsScaler,Normalizer - также неплохие стандартизаторы 





import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram # испортируем linkage,dendogram , которые нам понадобятся для создания иерархичскго графика
mergings = linkage(samples, method='complete') # полностью линкуем наши семплы 
# создем иерархо-грамму 
dendrogram(mergings, 
            labels=country_names,
            leaf_rotation=90,           
            leaf_font_size=6) 
plt.show()




from scipy.cluster.hierarchy import linkage # импортируем linkage,чтобы сгрупировать данные 
mergings = linkage(samples, method='complete')
from scipy.cluster.hierarchy import fcluster # импортируем fcluster,который помогает найти кластеры на дистанции
labels = fcluster(mergings, 15, criterion='distance') # выдает numpy array названий кластеров 
print(labels)



import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
mergings = linkage(samples,method = 'single') # когда мы используем другой метод группировки,то у нас получаются другие связи
dendrogram(mergings,labels = country_names,leaf_rotation = 90,leaf_font_size = 6)
plt.show()





import pandas as pd
pairs = pd.DataFrame({'labels': labels, 'countries': country_names})
print(pairs.sort_values('labels')) # смотрим какие кластеры,каким странам соответствуют


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE # импортируем TSNE, который в 2D показывает расположение данных относительно себя же 
model = TSNE(learning_rate=100) #  learning rate надо 'потыкать' от 50 и до 200
transformed = model.fit_transform(samples) # фитим и сразу трансформируем !!! TSNE по-другому не умеет,то есть надо данные собирать сразу вместе 
xs = transformed[:,0] # создаем ось x(первая колонка)
ys = transformed[:,1] # создаем ось y(вторая колонка)
plt.scatter(xs, ys, c=species) # сразу плотим график рассеиваниия + дополняем цветом 
plt.show()


# PCA - один из регуляторов
# 1.Decorelation(он убирает корреляцию - сводит к нулю,mean = 0)
# 2.Reducing demention(он убирает одну или несколько колонок,которые у нас играли какую-либо роль,не изменяя при этом ответа)
from sklearn.decomposition import PCA
model = PCA()
model.fit(samples)
transformed = model.transform(samples)
print(model.components_)
# этот код нам поможет понимать,сколько 'колонок' можно убрать не потеряв в качестве(показывает насколько колонки влияют(где какая дисперсия))
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xticks(features)
plt.ylabel('variance')
plt.xlabel('PCA feature')
plt.show()

# Тут мы уже сами выбираем количество столбцов,тем самым уменьшая их
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(samples)
transformed = pca.transform(samples)
print(transformed.shape)

# Тоже самое,что и PCA,просто работает по-другому(с векорами выборки,вместо матрицы )
from sklearn.decomposition import TruncatedSVD
model = TruncatedSVD(n_components=3)
model.fit(documents)  # documents is csr_matrix
TruncatedSVD(algorithm='randomized', ... )
transformed = model.transform(documents)

# NMF занимается тем же,что и PCA и TruncatedSVD
# Все данные должны быть неотрицательными 
# Все выходные данные 'интерпретируемые'
# Работает именно с строками,а не колонками(n_components - строки) 
# Используется в рекомендательных системах,Аудио,LED панелях
from sklearn.decomposition import NMF
model = NMF(n_components=2)
model.fit(samples)
nmf_features = model.transform(samples)
print(model.components_)


# так работает рекомендательная система 
import pandas as pd
from sklearn.preprocessing import normalize
norm_features = normalize(nmf_features)
df = pd.DataFrame(norm_features,index = titles)
article = df.loc['Cristiano Ronaldo']
similarities = df.dot(article)
print(similarities.nlargest())

# Перечесление всех точек на LED - панели
from sklearn.decomposition import NMF
model = NMF(n_components = 7) # всего их 7
features = model.fit_transform(samples)
for component in model.components_:
    show_as_image(component)
digit_features = features[0,:]
print(digit_features)
