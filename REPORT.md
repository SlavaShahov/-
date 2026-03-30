# Отчёт по Расчётно-графической работе  
**«Сетевые базы данных»**

**Вариант 8: Клуб единоборств**

**Выполнил:** студент группы ИП-214  
**Шахов В. Г.**

---

## 1. Описание баз данных

### 1.1. Инфологическое проектирование

**Таблица 1. Сущности и связи**

| Сущность 1          | Сущность 2       | Тип связи | Описание связи |
|---------------------|------------------|-----------|----------------|
| MartialArts        | Groups           | 1:M       | Один вид единоборств может иметь несколько групп |
| Trainers           | Groups           | 1:M       | Один тренер может вести несколько групп |
| Members            | Subscriptions    | 1:M       | Один участник может иметь несколько абонементов |
| Groups             | Subscriptions    | 1:M       | Одна группа может иметь несколько абонементов |

**Таблица 2. Атрибуты сущностей** (PostgreSQL)

| Сущность / Таблица | Атрибут                  | Поле в БД          | Тип данных      | Not Null | PK | FK | Доп. ограничения |
|--------------------|--------------------------|--------------------|-----------------|----------|----|----|------------------|
| MartialArts       | ID вида                  | art_id             | serial          | Yes      | Yes| -  | -                |
| MartialArts       | Название                 | name               | varchar(50)     | Yes      | -  | -  | UNIQUE           |
| Trainers          | ID тренера               | trainer_id         | serial          | Yes      | Yes| -  | -                |
| Trainers          | ФИО                      | full_name          | varchar(100)    | Yes      | -  | -  | -                |
| Trainers          | Стаж                     | experience_years   | int             | -        | -  | -  | ≥ 0              |
| Trainers          | Телефон                  | phone              | varchar(20)     | Yes      | -  | -  | UNIQUE           |
| Groups            | ID группы                | group_id           | serial          | Yes      | Yes| -  | -                |
| Groups            | Название                 | name               | varchar(100)    | Yes      | -  | -  | UNIQUE           |
| Groups            | Вид единоборств          | art_id             | int             | Yes      | -  | Yes| -                |
| Groups            | Тренер                   | trainer_id         | int             | -        | -  | Yes| -                |
| Groups            | Уровень                  | level              | varchar(20)     | -        | -  | -  | CHECK            |
| Members           | ID участника             | member_id          | serial          | Yes      | Yes| -  | -                |
| Members           | ФИО                      | full_name          | varchar(100)    | Yes      | -  | -  | -                |
| Members           | Дата рождения            | birth_date         | date            | Yes      | -  | -  | -                |
| Members           | Телефон                  | phone              | varchar(20)     | -        | -  | -  | UNIQUE           |
| Members           | Пол                      | gender             | char(1)         | -        | -  | -  | M/F              |
| Subscriptions     | ID абонемента            | subscription_id    | serial          | Yes      | Yes| -  | -                |
| Subscriptions     | Участник                 | member_id          | int             | Yes      | -  | Yes| -                |
| Subscriptions     | Группа                   | group_id           | int             | Yes      | -  | Yes| -                |
| Subscriptions     | Дата начала              | start_date         | date            | Yes      | -  | -  | -                |
| Subscriptions     | Дата окончания           | end_date           | date            | Yes      | -  | -  | > start_date     |
| Subscriptions     | Цена                     | price              | numeric(8,2)    | Yes      | -  | -  | > 0              |

(ER-диаграмма прилагается как Рисунок 1)

### 1.2. MongoDB

**Коллекции:**
- `events` — клубные события
- `feedback` — отзывы на тренеров
- `club_feedback` — отзывы на клуб

### 1.3. Redis (кэш)

| Ключ                          | Тип данных     | TTL    | Назначение |
|-------------------------------|----------------|--------|----------|
| `dashboard:stats`             | String (JSON)  | 120 с  | Статистика клуба (кол-во участников, групп, тренеров) |
| `dashboard:ratings:club_avg`  | String (JSON)  | 120 с  | Средний рейтинг клуба |
| `trainers:avg_ratings`        | Hash           | 120 с  | Средние рейтинги тренеров (trainer_id → JSON) |
| `dashboard:last_refresh`      | String         | —      | Время последнего обновления кэша |

---

## 2. Описание приложения

**Технологии:**
- Backend: Python 3 + Flask
- Базы данных: PostgreSQL (psycopg), MongoDB (pymongo), Redis (redis-py)
- Шаблоны: Jinja2
- Контейнеризация: Docker Compose

**Страницы приложения:**

- `/` — Главная (дашборд со статистикой и рейтингами)
- `/groups` — Список всех групп
- `/groups/<group_id>` — Состав конкретной группы
- `/trainers` — Список тренеров
- `/events` — Список событий (MongoDB)
- `/manage` — Управление (4 формы добавления)
- `/reviews` — Отзывы и средние оценки
- `/analytics` — Аналитика (SQL + MongoDB)
- `/status` — Статус всех сервисов
- `/api/status` — JSON API статуса

**Формы:**
- Добавление тренера, группы, участника в группу, события
- Добавление отзыва на тренера и на клуб

---

## 3. HTTP-методы серверной части

**GET /**  
Входные: отсутствуют  
Выходные: stats (members_count, groups_count, trainers_count), ratings (club_avg_rating, trainers_avg[]), from_cache, ratings_from_cache

**GET /groups**  
Входные: отсутствуют  
Выходные: rows[] (group_id, group_name, martial_art, trainer_name, members_count)

**GET /groups/<group_id>**  
Входные: group_id (в URL)  
Выходные: group_info + members_rows[]

**GET /trainers**  
Входные: отсутствуют  
Выходные: rows[] (trainer_name, experience_years, groups_count)

**GET /events**  
Входные: отсутствуют  
Выходные: rows[] (title, martial_art, event_date, participants_limit)

**GET /manage**  
Входные: отсутствуют  
Выходные: arts[], trainers[], groups[], postgres_ok, mongo_ok

**POST /manage** (action=add_trainer)  
Входные: full_name, phone, experience_years  
Выходные: message / error

**POST /manage** (action=add_group)  
Входные: name, art_id, trainer_id, level  
Выходные: message / error

**POST /manage** (action=add_member_to_group)  
Входные: member_full_name, birth_date, member_phone, gender, group_id, start_date, end_date, price  
Выходные: message / error

**POST /manage** (action=add_event)  
Входные: title, martial_art, event_date, participants_limit  
Выходные: message / error

**GET /reviews**  
Входные: отсутствуют  
Выходные: trainer_reviews[], club_reviews[], trainer_avg_rows[], club_avg_rating, trainers[]

**POST /reviews** (action=add_trainer_review)  
Входные: member_name, trainer_name, rating, comment  
Выходные: message / error

**POST /reviews** (action=add_club_review)  
Входные: author, rating, comment  
Выходные: message / error

**GET /analytics**  
Входные: отсутствуют  
Выходные: pg_rows[], mongo_rows[], club_avg_rating

**GET /status**  
Входные: отсутствуют  
Выходные: status{} (internet, postgres, mongo, redis)

**GET /api/status**  
Входные: отсутствуют  
Выходные: JSON {internet, postgres, mongo, redis}

---

## 4. Запросы к базам данных

- PostgreSQL: JOIN + GROUP BY (страницы `/groups`, `/trainers`, `/analytics`)
- MongoDB: агрегация `$group` + `$avg` + `$sort` (отзывы и рейтинги)
- Redis: кэширование с TTL 120 секунд (главная страница)

**Отказоустойчивость:** приложение продолжает работать при отключении любой из БД, показывая пользователю понятные сообщения.

**Дата:** 29 марта 2026 г.