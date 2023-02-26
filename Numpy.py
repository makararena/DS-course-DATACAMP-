# Питон все приводит к одному типу,поэтому  занимает мало места 
numpy_boolean_array = [[True, False], [True, True], [False, True]]
numpy_float_array = [1.9, 5.4, 8.8, 3.6, 3.2]

# Создает np array из нулей
np.zeros((3,5)) # Строки,колонки  изначально пишется в тупле  

# Рандомное float значение от 0 до 1,берет функцию рандом из библиотеки рандом,поэтому 2 рандома
np.random.random((2,4)) # Матрица 2 строки,4 колонки 

# Целые(int) значения от и до любого числа
np.arange((2,4)) # Целые значение от 2 до 4(не вкл)

# Создание тензора(3D array)
array_1_2D = np.array([[1, 2], [5, 7]])
array_2_2D = np.array([[8, 9], [5, 7]])
array_3_2D = np.array([[1, 2], [5, 7]])
array_3D = np.array([array_1_2D, array_2_2D, array_3_2D])


# Создание матрицы 3D матриц 
array_4D = np.array([array_A_3D, array_B_3D, array_C_3D, array_D_3D, array_E_3D,
                     array_F_3D, array_G_3D, array_H_3D, array_I_3D])

# Из матрицы создает обычный вектор 
array = np.array([[1, 2], [5, 7], [6, 6]])
array.flatten()

# Решейп(по названию понятно,что делает)
array = np.array([[1, 2], [5, 7], [6, 6]])
array.reshape((2, 3))

# У нас есть разные типы танных(чем меньше байтов(1 байт = 8 битов),тем легче весят и меньше занимают памяти)
# np.int64 --- np.int32 --- np.float64 --- np.float32 --- '<U12'(Строка)

float32_array = np.array([1.32, 5.78, 175.55], dtype=np.float32) # Создаем array и устанавливаем тип
float32_array.dtype # Смтрим тип данных

# Выделяем одно из чисел в обычном np.array
array[3]

# Тут уже из матрицы 1-row,2-column
sudoku_game[2, 4]

# Весь столбец матрицы
sudoku_game[:, 3]

# Всю строку матрицы 
sudoku_game[0]

# От 2 до 4 индекса(не вкл) в np.array
array[2:4]

# Тут уже действие происходит в матрице,2 - шаг
sudoku_game[3:6:2, 3:6:2]

# Сортим значения в матрице,по умолчанию axis = 1
np.sort(sudoku_game)

# Fancy indexing --- берем значения в первой колонке,которые делятбся на 2 без остатка и соединяем их с первой колонкой -> числа первой колонке,которые относятся ко второй колонке 
classroom_ids_and_sizes[:, 0][classroom_ids_and_sizes[:, 1] % 2 == 0]

# Возвращает индексы первой колонки,которые соответсвуют условию второй колонки 
np.where(classroom_ids_and_sizes[:, 1] % 2 == 0) 

# Заменяет числа,которые равны 0 на '',в sudoku_game 
np.where(sudoku_game == 0, "", sudoku_game)


# np.concatenate связывает np.array - и по нужному нам axis-у(думай о стрелочке(куда идет,там и связывается))
classroom_ids_and_sizes = np.array([[1, 22], [2, 21], [3, 27], [4, 26]])
grade_levels_and_teachers = np.array([[1, "James"], [1, "George"], [3,"Amy"],3, "Meehir"]])
np.concatenate((classroom_ids_and_sizes, grade_levels_and_teachers), axis=1)

# Удаляет любую строчку-колонку в classroom_data(тут максимально странно -> удаление происходит по y)
np.delete(classroom_data, 1, axis=1)

# Удаление без индекса приводит к тому,что ты просто удаляешь перое значение в первом столбце(1,1)
np.delete(classroom_data, 1)


# Методы агрегирования:
# .sum().min().max().mean().cumsum(),везде есть параметр axis = 0 или 1 

# Keepdims = True - означает,что мы сохраняем все в колонку или столбец в зависимости от того,что у нас было 
security_breaches.sum(axis=1, keepdims=True)

# как можно делать в numpy 
array = np.array([[1, 2, 3], [4, 5, 6]])
array + 3
array * 3
-----------------------------------------
array_a = np.array([[1, 2, 3], [4, 5, 6]])
array_b = np.array([[0, 1, 0], [1, 0, 1]])
array_a + array_b
----------------------------------------
array_a = np.array([[1, 2, 3], [4, 5, 6]])
array_b = np.array([[0, 1, 0], [1, 0, 1]])
array_a * array_b
-------------------------------------
array = np.array([[1, 2, 3], [4, 5, 6]])
array > 2
----------------------------------------


# Многие методы в обычном python не распространяються в numpy  и многого не умеют делать 
# np.vectorize решает эту проблему 
vectorized_len = np.vectorize(len)

# Также есть такая штука,как Broadcast(некоторые таблицы нельзя складывать друг с другом,а другие можно,поэтому,если,вдруг выдает ошибку,то пересмотри 3 главу курса)


# Загружаем файл .pny - самый удобный и быстрый формат для numpy
with open("logo.npy", "rb") as f:
    logo_rgb_array = np.load(f)
# Плотим фото 
plt.imshow(logo_rgb_array)
plt.show()

# Сохраняем файл в формате .npy
withopen("dark_logo.npy", "wb") as f:
    np.save(f, dark_logo_array)



# Меняем белый фон на ченый в задаче 
dark_logo_array = np.where(logo_rgb_array == 255, 50, logo_rgb_array)
plt.imshow(dark_logo_array)
plt.show()

# Хелпим методы 
help(np.unique)

# Хелпим атрибуты 
help(np.ndarray.flatten)

# Флипим фото(сразу флипиться по оси 0,1,2(нужна для тензоров(передняя матрица меняеться местами с третьей(именно с третьей,тк фото в формате RGB))))
flipped_logo = np.flip(logo_rgb_array)
plt.imshow(flipped_logo)
plt.show()

# Тут мы флипим толко одну ось(y)
flipped_rows_logo = np.flip(logo_rgb_array, axis=0)

# Тут флипим только цвет 
flipped_colors_logo = np.flip(logo_rgb_array, axis=2)

# Тут и первую и вторую
flipped_except_colors_logo = np.flip(logo_rgb_array, axis=(0, 1))

# Меняем строки и колонки местами 
np.transpose(array)

# Тут меняем местами тоже только колонки и строки(фото у нас повернеться на 90 градусов )
transposed_logo = np.transpose(logo_rgb_array, axes=(1, 0, 2)

# Легкий способ разобрать тензор(состоязий из трех частей) на несколько array-ев 
red_array, green_array, blue_array = np.split(rgb, 3, axis=2)

# Тут уже легкий способ соединение трех матриц в один тензор
stacked_rgb = np.stack([red_array, green_array, blue_array], axis=2)
