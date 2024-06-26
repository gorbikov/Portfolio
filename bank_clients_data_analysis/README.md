# Анализ клиентской базы банка
Из банка стали уходить клиенты. Каждый месяц. Немного, но заметно. Банковские маркетологи посчитали: сохранять текущих клиентов дешевле, чем привлекать новых. 

## Данные
Для обучения предоставлены исторические данные о поведении клиентов и расторжении договоров с банком:
* `RowNumber` — индекс строки в данных;
* `CustomerId` — уникальный идентификатор клиента;
* `Surname` — фамилия;
* `CreditScore` — кредитный рейтинг;
* `Geography` — страна проживания;
* `Gender` — пол;
* `Age` — возраст;
* `Tenure` — сколько лет человек является клиентом банка;
* `Balance` — баланс на счёте;
* `NumOfProducts` — количество продуктов банка, используемых клиентом;
* `HasCrCard` — наличие кредитной карты;
* `IsActiveMember` — активность клиента;
* `EstimatedSalary` — предполагаемая зарплата;
* `Exited` — факт ухода клиента.

## Задача
Необходимо спрогнозировать, уйдёт клиент из банка в ближайшее время или нет и построить модель с предельно большим значением F1-меры. Необходимо довести метрику до 0.59.

## Используемые библиотеки
*pandas*, *numpy*, *matplotlib*, *seaborn*, *sklearn*

## Выводы
* Построен ряд моделей (LR, DTC, RFC), подобраны оптимальные параметры. Лучшие результаты по F1 score показала модель RFC. Эта модель была выбрана за базовую.
* Для улучшения результатов базовой модели проверен ряд гипотез. Проверка показала, что качество модели по метрике F1 score можно улучшить за счёт устранения дисбаланса в целевом столбце (upsampling). Подобран оптимальный threshold.
* **На тестовой выборке достигнуты результаты выше целевых. Итоговый F1 score при стандартной threshold: 0.62. Итоговая оценка ROC-AUC на тестовой выборке составила 0.85. Работа завершена.**