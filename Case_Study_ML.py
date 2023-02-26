# Если вдруг нам нужно будет тренировать модель,то она работает именно с цифрами,а не буквами 
# Поэтому мы можем буквенные переменные a/b перевести в категориальные 0/0/1,но не показать их
from xml.etree.ElementPath import prepare_predicate


sample_df.label = sample_df.label.astype('category')


# Функция pd.get_dummies  помогает нам именно показать эту бинарную разницу 
# Тут у нас будет уже совершенно другая таблица с 0/1
dummies = pd.get_dummies(sample_df[['label']], prefix_sep='_')

# Лямбда - тоже самое,что и def функция,только пишется в одну строчку 
categorize_label = lambda x: x.astype('category')

# Тут мы к колонке 'label' применяем нашу лямбду,которая переделывает всю колнку в категориальную переменную
# Axis = 0 нам нужен обязательно,потому что мы все равно действуем на строки,чтобы поменять их в категории( axis = 0 ----- row )
sample_df.label = sample_df[['label']].apply(categorize_label, axis=0)

# чтобы поменять все колнки нам просто label ннадо поменять на LABELS 
df[LABELS] = df[LABELS].apply(categorize_label,axis = 0)

# Смотрим типы всех колонок
print(df[LABELS].dtypes)

# Логистическая ошибка очень хороошо реагирует на увереность AI в неправильном ответе 
logloss = -1/N * E(yi log(pi)) + (1 - yi)log(1 - pi)
Actual Value : y = {0,1} # То есть мы можем выбрать от 0 до 1 и при этом в p value уже вносим вероятность,что наше число - 1
Prediction(Prediction that value is 1):p




# Если вдруг я буду учавствовать в соревнованиях,то всегда начинают с самой простой модели и смотрят результаты,насколько все сложно
# Поэтому мы будем использовать Multi-Class Logistic Regression
# Но в нашем случае есть проблемы с тем,что данных слишком мало, и если мы будем делить на train/test,то у нас могут появиться выскочки в test split,что значительно ухудшит результаты 
# Поэтому у нас есть два решения этой проблемы 
# 1.StratifiedShuffleSplit , но оно работает,только если у нас есть одна target variable 
# 2.multilabel_train_test_split() , она уже работает с несколькими переменными 

data_to_train = df[NUMERIC_COLUMNS].fillna(-1000) # филим NaN с -1000,чтобы у нас получились именно 'выскочки'
labels_to_use = pd.get_dummies(df[LABELS]) #  переводим в категориальные переменные 
X_train, X_test, y_train, y_test = multilabel_train_test_split(data_to_train,labels_to_use,size=0.2, seed=123) # Делим 
from sklearn.linear_model import LogisticRegression 
from sklearn.multiclass import OneVsRestClassifier # К каждой колонке применяет отдельную регрессию
clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X_train, y_train)

# Тут мы уже начинаем предиктить 
# clf.predict отличается от clf.predict_proba в том,что predict выбирает определенные числа(ответы),а predict_proba отображает именно вероятности 
holdout = pd.read_csv('HoldoutData.csv', index_col=0)
holdout = holdout[NUMERIC_COLUMNS].fillna(-1000)
predictions = clf.predict_proba(holdout)


# Переводим все в датафрейм и сsv и после это скорим результаты
prediction_df = pd.DataFrame(columns=pd.get_dummies(df[LABELS],prefix_sep='__').columns,index=holdout.index,data=predictions)
prediction_df.to_csv('predictions.csv')
score = score_submission(pred_path='predictions.csv')


------------------------------------------------------------
# Введение в NLP
# 1.Мы можем разделять приложения на слова и токенизировать каждое слово и после это считать сколько слов было в том или ином месте,
# и сколько всего их в общем
# также можем разделять слова с тире 
# Но самая главная проблема там - мы не видим порядка,поэтому blue,not red - тоже самое,что и red,not blue 
# 2.Мы можем разделять слова на грамовки(n_grams) - 1-gram,2-gram,3-gram и тп


# СоuntVectorizer работает именно,как первый случай э
# 1. Токенизирует все строки 
# 2. Создает 'словарь'
# 3. Считает количество каждого слова в словаре
from sklearn.feature_extraction.text import CountVectorizer
TOKENS_BASIC = '\\\\S+(?=\\\\s+)' # Токенный пробел,надо подробно про это почитать 
df.Program_Description.fillna('', inplace=True) # Меняем все NaN на ничего
vec_basic = CountVectorizer(token_pattern=TOKENS_BASIC) # Создаем модель,внутри нее токен 
vec_basic.fit(df.Program_Description) # Фитим 
msg = 'There are {} tokens in Program_Description if tokens are any non-whitespace'
print(msg.format(len(vec_basic.get_feature_names())))# Принтим 


# Выводит поледние строки 
df.tail(n = x) 

---------------------------------------------------------------
# Imputer заполняет NaN
from sklearn.preprocessing import Imputer
# Импортируем CountVectorizer,который токенизирет слова 
from sklearn.feature_extraction.text import CountVectorizer
# Импортируем Pipeline,который делает так,чтобы мы могли упростить наш код путем 'шагов'
# Самое важное - Pipeline работает только с туплами,нужны примерные названия функций и классификаторов,регрессоров 
from sklearn.pipeline import Pipeline 
# Импортируем Логистическую регрессию
from sklearn.linear_model import LogisticRegression
# Импортируем штуку,которая позволяет создавать копии классификаторов и каждый классификатор применяет к отдельному классу 
from sklearn.multiclass import OneVsRestClassifier
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing', 'text']], pd.get_dummies(
                                        sample_df['label']), random_state=2)
# Функция,которая приводит текстовый класс и int class to one class
from sklearn.preprocessing import FunctionTransformer
# Функция,которая позволяет связывать текстовый класс и номерной класс 
from sklearn.pipeline import FeatureUnion
get_text_data = FunctionTransformer(combine_text_columns,validate=False) #validate = false,нужен для того,чтобы python не проверял колонки на NaN
get_numeric_data = FunctionTransformer(lambda x:x[NUMERIC_COLUMNS], validate=False) # Выбираем только номерные колонки,также с validate 
# Создаем Pipeline,который работает так:
# 1.FeatureUnion 
#    ---- 1.Pipeline 
#            ------ 1. get numeric data 
#            ------ 2. Imputer()
#    ---- 2.Pipeline 
#            ------ 1. get text data
#            ------ 2. CountVectorizer()
# 2.OneVsRestClassifier(LOgisticRegression())
pl = Pipeline([
                 ('union', FeatureUnion([                     
                    ('numeric_features', Pipeline([                         
                        ('selector', get_numeric_data),                                              
                        ('imputer', Imputer())                     
                        ])),                     
                    ('text_features', Pipeline([                         
                        ('selector', get_text_data),                         
                        ('vectorizer', CountVectorizer())                     
                        ]))                  
                    ])             
                ),             
                ('clf', OneVsRestClassifier(LogisticRegression()))         
            ])
pl.fit(X_train, y_train)



LABELS = ['Function', 'Use', 'Sharing', 'Reporting', 'Student_Type','Position_Type', 'Object_Type',  'Pre_K', 'Operating_Status']
NON_LABELS = [c for c in df.columns if c notin LABELS]




# Так работает функция combine_text_columns 
def combine_text_columns(data_frame, to_drop=NUMERIC_COLUMNS + LABELS):
    """ converts all text in each row of data_frame to single vector """


    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop,axis = 1)
    
    text_data.fillna('',inplace = True)
    
    return text_data.apply(lambda x: " ".join(x), axis=1)


-----------------------------------------------------------------------------------------------------------------------
# так работают n-graммы 
vec = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC,ngram_range=(1, 2))

# Это все надо для того,чтобы понимать именно место текста в пространстве,тк English teacher for 2nd grade 
# и 2nd grade - budget for english teacher меняет именно смысл 
# Штука,которая работает,как polynominal features - берет таблицу и высчитывает B3X3 
# Только проблема  том,что PolynominalFeatures не работают со sparse  matrix,а SparceInteractions работает
SparseInteractions(degree=2).fit_transform(x).toarray()

# Хеширование работает по абсолютно такому же сценарию,как и CountVectorizer,только весь текст переводит в хеши,тем самым уменьшая память и уменьшая(немного) производительность
from sklearn.feature_extraction.text import HashingVectorizer
vec = HashingVectorizer(norm=None,non_negative=True,token_pattern=TOKENS_ALPHANUMERIC,ngram_range=(1, 2))

