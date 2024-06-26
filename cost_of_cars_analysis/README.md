# Определение стоимости автомобилей
Сервис по продаже автомобилей с пробегом разрабатывает приложение, чтобы привлечь новых клиентов. Заказчик хочет, чтобы в нём можно было узнать рыночную стоимость своего автомобиля.

Критерии, которые важны заказчику:
* качество предсказания
* время обучения модели
* время предсказания модели

## Данные
В распоряжении имеются данные о технических характеристиках, комплектации и ценах других автомобилей от сервиса по продаже автомобилей:
* `DateCrawled` — дата скачивания анкеты из базы;
* `VehicleType` — тип автомобильного кузова;
* `RegistrationYear` — год регистрации автомобиля;
* `Gearbox` — тип коробки передач;
* `Power` — мощность (л. с.);
* `Model` — модель автомобиля;
* `Kilometer` — пробег (км);
* `RegistrationMonth` — месяц регистрации автомобиля;
* `FuelType` — тип топлива;
* `Brand` — марка автомобиля;
* `Repaired` — была машина в ремонте или нет;
* `DateCreated` — дата создания анкеты;
* `NumberOfPictures` — количество фотографий автомобиля;
* `PostalCode` — почтовый индекс владельца анкеты (пользователя);
* `LastSeen` — дата последней активности пользователя;
* `Price` — цена (евро).

## Задача
Необходимо построить модель, которая умеет определять рыночную стоимость автомобиля. Значение метрики RMSE должно быть меньше 2500.

**Тип задачи:** регрессия.

## Используемые библиотеки
*pandas*, *numpy*, *scipy*, *matplotlib*, *seaborn*, *sklearn*, *lightgbm*

## Выводы
* Данные подготовлены для обучения. Заполнены пропуски, категориальные переменные переведены в числовые методом target encoding.
* Исследованы три типа моделей DTR, RFR и LGBM. Для каждой модели подобраны лучшие параметры. Лучшие результаты по метрике RMSE показала модель RFR.
* На тестовой выборке результат этой модели: RMSE ~ 1750, время обучения около 22 секунд, время расчёта предсказания ~600 мс.
* Для сравнения результаты LGBM: RMSE ~ 1810, время обучения около 1 секунд, время расчёта предсказания - менее 150 мс.
* **Итоговые метрики точности для обеих моделей не превышают целевые 2500 RMSE. Для заказчика подготовлены рекомендации по выбору из 2 имеющихся альтернатив. Работа завершена.**