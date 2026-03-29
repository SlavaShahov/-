# Отчет по РГР (Сетевые БД)

## Тема
Клуб единоборств.

## 1. Описание баз данных

### 1.1 PostgreSQL (основная БД)

Сущности:
- `MartialArts`
- `Trainers`
- `Groups`
- `Members`
- `Subscriptions`
- `Attendances`

Связи:
- `MartialArts (1) -> (M) Groups`
- `Trainers (1) -> (M) Groups`
- `Members (1) -> (M) Subscriptions`
- `Groups (1) -> (M) Subscriptions`
- `Subscriptions (1) -> (M) Attendances`

Ключи и ограничения:
- первичные ключи `SERIAL` во всех таблицах;
- внешние ключи между связанными таблицами;
- `UNIQUE` для названий направлений и телефонов;
- `CHECK` для возраста, пола, уровней групп, стажа и цены.

Нормальная форма:
- таблицы находятся в 3НФ, атрибуты зависят от ключа и не содержат транзитивных зависимостей.

### 1.2 MongoDB

Коллекции:
- `events` (клубные события)
- `feedback` (отзывы о тренерах)

Поля в `events`:
- `title`, `martial_art`, `event_date`, `participants_limit`

Поля в `feedback`:
- `member_name`, `trainer_name`, `rating`, `comment`, `created_at`

### 1.3 Redis

Ключи:
- `dashboard:stats` (JSON со сводной статистикой, ключ с TTL)
- `dashboard:last_refresh` (время последнего обновления кеша)

## 2. Описание приложения

Технологии:
- Python + Flask
- PostgreSQL (`psycopg`)
- MongoDB (`pymongo`)
- Redis (`redis-py`)
- Docker Compose

Страницы:
- `/` - главная сводка
- `/groups` - список групп и количество участников
- `/trainers` - тренеры и их нагрузка
- `/events` - события из MongoDB
- `/analytics` - аналитика по PostgreSQL и MongoDB
- `/status` - мониторинг доступности интернета и БД

## 3. HTTP методы серверной части

- `GET /`
  - Входные: нет
  - Выходные: HTML главной страницы со статистикой

- `GET /groups`
  - Входные: нет
  - Выходные: HTML с данными по группам (JOIN + GROUP BY)

- `GET /trainers`
  - Входные: нет
  - Выходные: HTML с данными по тренерам (JOIN + GROUP BY)

- `GET /events`
  - Входные: нет
  - Выходные: HTML с событиями из MongoDB

- `GET /analytics`
  - Входные: нет
  - Выходные: HTML с SQL-аналитикой и MongoDB aggregation

- `GET /status`
  - Входные: нет
  - Выходные: HTML со статусом доступности сервисов

- `GET /api/status`
  - Входные: нет
  - Выходные: JSON, например:
    - `internet: true/false`
    - `postgres: true/false`
    - `mongo: true/false`
    - `redis: true/false`

## 4. Запросы к БД (выполнены в коде)

PostgreSQL:
- на странице `/groups` используется `JOIN` + `GROUP BY` для подсчета участников в группе;
- на странице `/trainers` используется `LEFT JOIN` + `GROUP BY` для нагрузки тренеров;
- на странице `/analytics` вычисляется средняя цена абонемента по виду спорта (`JOIN` + `GROUP BY` + `AVG`).

MongoDB:
- на странице `/analytics` используется `aggregate` с `$group` и `$sort` для среднего рейтинга тренеров.

Redis:
- на странице `/` используется кеш `dashboard:stats` через `SETEX` (TTL);
- ключ `dashboard:last_refresh` хранит время обновления.

## 5. Отказоустойчивость

Предусмотрены сценарии:
- отключение интернета;
- отключение PostgreSQL;
- отключение MongoDB;
- отключение Redis.

При отказе отдельного сервиса:
- приложение не падает;
- соответствующая страница показывает предупреждение и частичные данные;
- статус можно проверить на `/status` или `/api/status`.
