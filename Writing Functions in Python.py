#Docstring formats 
#-Google Style 
#-Numpydoc 
#-reStructuredText
#-EpyText

# Google Style 
def function(arg_1, arg_2=42):
"""Description of what the function does.  

    Args:
        arg_1 (str): ......
        arg_2 (int, optional): .....

    Returns:
        bool: ......

    Raises:.......  

    Notes: ....    """


# Numpydoc 

"""  Description of what the function does.  

    Parameters 
    ----------  
    arg_1 : expected type of arg_1    
        Description of arg_1.  
    arg_2 : int, optional    Write optional when an argument has a default value.    Default=42.  
    

    Returns  
    -------
     The type of the return value    
     Can include a description of the return value.     
     Replace "Returns" with "Yields" if this function is a generator.
"""

# как посмотреть документацию функции 
def the_answer():
    return 42
print(the_answer.__doc__)

# как посмотреть документацию функции 2.0
import inspect
print(inspect.getdoc(the_answer)




#Основные правила писания кода 
1.не повторяйся(лучше напиши функцию,чем писать дохера кода,причем максимально похожего)
2.Не сильно услажнять код,то есть не писать одну дико сложную функцию,а лучше написать две маленбкие,котоые взаимодействуют между собой

Advantages of doing one thing 
    The code becomes:
    -More flexible
    -More easily understood 
    -Simpler to test 
    -Simpler to debug 
    -Easier to change



Не изменяемые объекты:
-int 
-str
-float 
-bool 
-bytes
-tuple 
-frozenset 
-None 

Изменяемые объекты:
-list 
-dict 
-set 
-bytearray
-objects
-functions
almost everything else!

# Надо быть очень аккуратным с именяемыми объектами,тк если мы будем использовать функцию foo() два раза,то создасться array[1,1]
def foo(var=[]):
  var.append(1)
  return var
foo()

# Лучше всего использовать функцию именно так,тогда у нас всегда будет array [1]
def foo(var=None):
    if var is None:
        var = []  
    var.append(1)
    return var 
foo()



# функция with open делает три вещи(1.Достает все,открывая файл  2.позволяет нам запустить любой код внутри этой функции и перезаписать его во что-то
# 3.Убирает все,что у нас осталось когда мы уже не работаем в этом файле) 
# Штука as позволяет нам сохранить все в какую-либо переменную 
with open('my_file.txt') as my_file:
      text = my_file.read()  
      length = len(text)
print('The file is {} characters long'.format(length))



@contextlib.contextmanager #декоратор

def my_context():
      print('hello')
      yield 42 #эта штука нажна при использовании функции,она начинает хранить в себе значение определенной переменной или числа 
      print('goodbye')

with my_context() as foo:
      print('foo is {}'.format(foo))



# пример использование yield 
@contextlib.contextmanager
def open_read_only(filename):
  """Open a file in read-only mode.

  Args:
    filename (str): The location of the file to read

  Yields:
    file object
  """
  read_only_file = open(filename, mode='r')
  # Yield read_only_file so it can be assigned to my_file
  yield read_only_file 
  # Close read_only_file
  read_only_file.close()

with open_read_only('my_file.txt') as my_file:
  print(my_file.read())



# Мы перезаписываем из одного файла данные в другой файл,но делаем это построчно с помощью with и as 
def copy(src, dst):
    """Copy the contents of one file to another.
    
  Args:    
    src (str): File name of the file to be copied.
    dst (str): Where to write the new file.
    """
    # Open both files
     with open(src) as f_src:
        with open(dst, 'w') as f_dst:
            # Read and write each line, one at a time
            for line in f_src:
                f_dst.write(line)



#Делаем так,чтобы функция работала в любом случае 
try:
    # что-то типо if 
except:
    # что-то типо elif 
finally:
    # что-то типо else


#Пример использование try и finally(тут проблема в том,что если мы оставим только одно значение yield,то connection откроется,займет нашу память и потом не закроется,если yield не сработает,поэтому лучше всего использовать finally и except)
def get_printer(ip):  
    p = connect_to_printer(ip)
    
    try:
        yield
    finally:
        p.disconnect()
        print('disconnected from printer')

doc = {'text': 'This is my text.'}

with get_printer('10.0.34.111') as printer:
      printer.print_page(doc['txt'])


# штуки,которые наверное как-то мажно использовать,но все используют Open/Close 
Open-Close
Lock-Release
Change-Reset
Enter-Exit
Start-Stop
Setup-Teardown
Connect-Disconnect



# Коды,которые показывают,что функции-такие же объекты,как и все остальные 
PrintyMcPrintface = print
PrintyMcPrintface('Python is awesome!')

# функции можно добалять в листы 
list_of_functions = [my_function, open, print]
list_of_functions[2]('I am printing with an element of a list!')

#Даже можно добавлять в словари 
dict_of_functions = {'func1': my_function,'func2': open,'func3': print}
dict_of_functions['func3']('I am printing with a value of a dict!')



# код,который проверяет,если ли в функции """объяснение""" и потом эту функцию можно применять к другим функциям 
defhas_docstring(func):
    """Check to see if the function   
    `func` has a docstring.  
        Args:    
            func (callable): A function.  
        Returns:    bool  """
return func.__doc__ is not None

has_docstring(print)



# Вложенные функции(требуют меньше кода и более читаемые)
def foo(x, y):
    def in_range(v):
        return v > 4 and v < 10 
        
    if in_range(x) and in_range(y):
        print(x * y)


# какие бывают области,в которых функция ищет значения 
Builtin -> global -> nonlocal -> local
# функция global переводит любое значение в глобальную обсласть 
# функция nonlocal сохраняет значение и во внешней функции(была внутренняя и вешняя,во нутренней устанавливаем и потом во внешней такое же значение)

# использование global
x = 7
def foo():
    global x
    x = 42
    print(x)

foo()

#использование nonlocal
def foo():
      x = 10 
      def bar():
          nonlocal x
        x = 200    
        print(x)  
    bar()  
    print(x)
foo()


#: “замыкание (closure) в программировании — это функция,
#  в теле которой присутствуют ссылки на переменные, объявленные вне тела этой функции в окружающем коде и не являющиеся ее параметрами.”
# Если просто,то замыкания озволяют нам сохранять в внешнюю функцию какое-то значение и оно будет там висеть еще очень долго,потому что значение замыкается и даже,если мы ее перепишем в другую переменную,то значение сохранится 
# и даже,если мы удалим a ,то оно все равно сохранится в нашей функции 
def foo():
      a = 5
    def bar():
        print(a)
    return bar
func = foo()
func()
и выведет 5 

# помогает посмотреть длину значений функции
len(func.__closure__)
#удаляет что-то
del(x)
# Узнаем тип замыкающихся значений 
type(func.__closure__)
# находим какой-либо элемент в замыкающихся значениях
func.__closure__[0].cell_content



#Декоратор - это штука,которая может изменить значение input в нашу функцию,значение output или само значение функции 
@double_args-> пример декоратора (он может быть уже в базе или мы его сами можем создать из функции.Главное-убрать ())
def multiply(a, b):
    return a * b
multiply(1, 5)


# синтаксис декоратора Пример-Basic 
def double_args(func):
    def wrapper(a, b):
        return func(a * 2, b * 2)
    return wrapper
    
def multiply(a, b):
    return a * b
    
multiply = double_args(multiply)
multiply(1, 5)

#Тот же самый пример,только уровень-Intermediate 
def double_args(func):
    def wrapper(a, b):
        return func(a * 2, b * 2)
    return wrapper
    
@double_args 
def multiply(a, b):
    return a * b
    
multiply(1, 5)
