# example_step.py

# 1
import some_lib1
import some_lib2

# 2
def helper_function1():
    # код необходимой функции #
    return

def helper_function2():
    # код необходимой функции #
    return

# 3
def main_function():
    helper_function1()
    helper_function2()

    # --- 3.1 Загрузка файла с гиперпараметрами
    import yaml
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd) 

    # --- 

    # --- 3.2 Загрузка результатов предыдущих шагов
    import pandas as pd
    import joblib

    # например, загрузка данных
    data = pd.read_csv('data-directory/data-file')

    # или загрузка модели
    with open('model-directory/model-file', 'rb') as fd:
        model = joblib.load(fd)    
    # --- 

    # --- 3.3 код необходимой функции #
    # ...
    # --- 

    # --- Сохранение результатов текущего шага
    import pandas as pd
    import joblib

    # например
    os.makedirs('data-directory', exist_ok=True) # создание директории, если её ещё нет
    data.to_csv('path/to/data/file')

    # или
    os.makedirs('model-directory', exist_ok=True) # создание директории, если её ещё нет
    with open('path/to/model/file', 'wb') as fd:
        joblib.dump(model, fd)
    # --- 


    return

# 4
if __name__ == '__main__':
    main_function()