# RGR: Клуб единоборств (Сетевые БД)

Веб-приложение на Python (`Flask`) для предметной области "Клуб единоборств" с использованием трех СУБД:

- `PostgreSQL` - основная реляционная БД (группы, тренеры, участники, абонементы, посещения)
- `MongoDB` - документная БД (события и отзывы)
- `Redis` - кеш с TTL

## Технологии

- Python 3.12
- Flask
- psycopg
- pymongo
- redis-py
- Docker Compose

## Функциональность

Приложение содержит страницы:

- `/` - главная страница (сводная статистика + рейтинги клуба/тренеров из Redis)
- `/groups` - список групп
- `/groups/<group_id>` - состав конкретной группы
- `/trainers` - тренеры и количество групп
- `/events` - список событий клуба
- `/manage` - управление данными (добавление тренеров, групп, участников, событий)
- `/reviews` - отзывы на тренеров и клуб, расчет средних рейтингов
- `/analytics` - SQL/MongoDB аналитика
- `/status` - статус доступности интернет/БД
- `/api/status` - API-статус (JSON)

## Соответствие требованиям РГР

- Клиент и сервер взаимодействуют по HTTP.
- Реализовано больше 5 страниц.
- PostgreSQL содержит 6 таблиц предметной области с PK/FK/ограничениями.
- MongoDB содержит не менее 2 коллекций (`events`, `feedback`, `club_feedback`).
- Redis содержит ключи с данными из БД и TTL.
- Есть запросы к PostgreSQL с `JOIN` и `GROUP BY`.
- Есть агрегации MongoDB (`$group`, `$avg`, `$sort`).
- Есть Redis-ключи с TTL через `SETEX`.

## Redis-ключи

Основные ключи кеша:

- `dashboard:stats` - сводная статистика (TTL)
- `dashboard:last_refresh` - время последнего обновления
- `dashboard:ratings:club_avg` - средний рейтинг клуба (TTL)
- `dashboard:ratings:trainers_avg` - средние рейтинги тренеров (TTL)

## Запуск

Из корня проекта:

```bash
docker compose up --build
```

Запуск в фоне:

```bash
docker compose up --build -d
```

Открыть в браузере:

- [http://localhost:8000](http://localhost:8000)

## Управление сервисами БД

Остановить сервис:

```bash
docker compose stop postgres
docker compose stop mongo
docker compose stop redis
```

Запустить сервис обратно:

```bash
docker compose start postgres
docker compose start mongo
docker compose start redis
```

Посмотреть статус контейнеров:

```bash
docker compose ps
```

## Поведение при отказах

Приложение работает в деградирующем режиме:

- при недоступности `PostgreSQL` блокируется добавление SQL-данных (тренеры/группы/участники),
- при недоступности `MongoDB` блокируется добавление событий/отзывов,
- при недоступности `Redis` приложение продолжает работу без кеша,
- пользователю выводятся понятные предупреждения вместо технических stack trace.

## Инициализация данных

При первом запуске Docker Compose автоматически применяются:

- `scripts/init_postgres.sql` - создание таблиц и тестовые данные PostgreSQL
- `scripts/init_mongo.js` - тестовые коллекции и документы MongoDB

## Структура проекта

- `app/app.py` - Flask-приложение, маршруты, работа с БД и кешем
- `app/templates/` - HTML-шаблоны страниц
- `app/static/styles.css` - стили интерфейса
- `scripts/init_postgres.sql` - SQL-схема и начальные данные
- `scripts/init_mongo.js` - начальные документы MongoDB
- `docker-compose.yml` - оркестрация контейнеров
- `Dockerfile` - образ веб-приложения
- `REPORT.md` - заготовка отчета по РГР
