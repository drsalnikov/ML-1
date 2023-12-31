# ML-1
Homework ML-1

# Что было сделано
1. Обработаны признаки как категориальные так и вещественные. Масштабирование признаков и OneHot кодирование категориальных признаков. 
2. Проведена визуализация: попарные распределения признаков, тепловая карта корреляции признаков.  
3. Созданы модели: Линейная регрессия, Lasso-регрессия с регуляризацией, ElasticNet регрессия с двумя регуляризаторами.
4. Подобраны гиперпараметры через CridSearchCV. 
5. Посчитана кастомная метрика -- среди всех предсказанных цен на авто посчитана доля предиктов, отличающихся от реальных цен на эти авто не более чем на 10%
6. Создан сервис при помощи FastApi

# Какие результаты
1. Изучены правила составления базового EDA по данным. 
2. Подобраны гиперпараметры, и изучены базовые метрики для сравнения результатов полученных моделей. 

# Что дало наибольший буст в качестве
Наибольший буст в качестве дает качественная обработка признаков, подбор гиперпарраметров и общее понимание бизнес-требования. 

# Что сделать не вышло и почему
1. При получении новой категории, которая не была заложена в тренировочные данные, нужно переобучать модель. 
Подход требует не только инструментов создания REST API, но и пайплайнов/сервисов для переобучения модели. 
2. Не удалось выполнить все доп задания, вероятно из-за не хватки времени или не достаточной осведомленности в вопросе ML.


# Работа сервиса 

 * Запуск ```uvicorn index:app --reload```
 * Скриншоты

1. Swagger
![Alt text](/image/image.png)

2. ```/predict_item```
![Alt text](/image/image2.png)

3. ```/predict_items```
![Alt text](/image/image3.png)

![Alt text](/image/image4.png)

4. Загрузка файла из .csv ```/uploadfile*```

![Alt text](/image/image5.png)

![Alt text](/image/image6.png)
