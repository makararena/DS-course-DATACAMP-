def square(value):
    """Return the square of a value."""
        new_value = value ** 2
        return new_value
# создание своей функции и также возвращаем new_value 

def raise_to_power(value1, value2):
    """Raise value1 to the power of value2."""
        new_value = value1 ** value2
        return new_value
# использование двух функций 


def mod2plus5(x1, x2, x3):
    """Returns the remainder plus 5 of three values."""
        def inner(x):"""Returns the remainder plus 5 of a value."""
                return x % 2 + 5
            return (inner(x1), inner(x2), inner(x3))

print(mod2plus5(1, 2, 3))
# использование внутренней функции



def raise_val(n):
    """Return the inner function."""

    def inner(x):"""Raise x to the power of n.""" 
            raised = x ** n
            return raised
    return inner
square = raise_val(2)
cube = raise_val(3)
print(square(2), cube(4))

4 
64
# использование внутренней функции 2.0

def outer():
    """Prints the value of n."""
    n = 1 
        
    def inner():
        nonlocal n 
         n = 2
         print(n)
inner()
print(n)
outer()

2
2
# использование nonlocal 
#nonlocal по сути образует промежуточное звено
#между глобальной и локальной областью.


defpower(number, pow=1):
    """Raise number to the power of pow."""
    new_value = number ** pow
    return new_value
#приравнивание аргумента к какому-то числу -> если этого аргумента не будет,то подставляется это число 

def add_all(*args):
#можно использовать любое количество аргументов 



defprint_all(**kwargs):
    """Print out key-value pairs in **kwargs."""
    
    # Print out the key-value pairs 
for key, value in kwargs.items():
        print(key + \": \" + value)
print_all(name="dumbledore", job="headmaster")
# использование key/value в переменной 

raise_to_power = lambda x, y: x ** y
raise_to_power(2, 3)

8
# использование лябды(использование функции,но в строчку)

map(func, seq)
# применяет любую функцию к последовательности 

nums = [48, 6, 9, 21, 1]
square_all = map(lambda num: num ** 2, nums)
print(list(square_all))

[2304, 36, 81, 441, 1]
# пример 

defsqrt(x):
    """Returns the square root of a number."""
    if x < 0:
        raise ValueError('x must be non-negative')#также можно использовать 
    try:
        return x ** 0.5
    except (TypeError):
        print('x must be an int or float')sqrt(4)
# вывод самой простой ошибки (TypeError)
