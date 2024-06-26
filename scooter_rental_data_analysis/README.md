# Анализ данных сервиса аренды самокатов GoFast
Проект по анализу данных популярного сервиса аренды самокатов GoFast. Чтобы совершать поездки по городу, пользователи сервиса GoFast пользуются мобильным приложением. Сервисом можно пользоваться как с подпиской, так и без неё.

## Данные
 Данные содержат информацию о некоторых пользователях из нескольких городов, а также об их поездках:
* `user_id`	- уникальный идентификатор пользователя;
* `name` - имя пользователя;
* `age` - возраст;
* `city` - город;
* `subscription_type` - тип подписки (free, ultra).

## Задача
Провести анализ данных и проверить ряд гипотез, которые могут помочь бизнесу вырасти.

## Используемые библиотеки
*pandas*, *numpy*, *scipy*, *matplotlib*

## Выводы
* Проведена очистка и подготовка данных.
* Проведён анализ различий между премиальными пользователями и пользователями с обычными аккаунтами.
* Проведена статистическая проверка ряда гипотез, связанных с размером трат пользователей сервиса и расстояниями, которые пользователи проезжают за одну поездку.
* **Статистически обоснован вывод о том, что премиальные подписчики являются более выгодными клиентами для компании, чем бесплатные. Работа завершена.**
