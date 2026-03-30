# Клуб единоборств — Расчётно-графическая работа

**Предмет:** Сетевые базы данных  
**Вариант:** 8  

Веб-приложение на Flask с использованием трёх баз данных:  
**PostgreSQL + MongoDB + Redis**

---

## Технологии

- **Python 3.12** + **Flask**
- **PostgreSQL** (`psycopg`)
- **MongoDB** (`pymongo`)
- **Redis** (`redis-py`)
- **Docker Compose**

---

## Функциональность

Приложение содержит следующие страницы:

- **`/`** — Главная (дашборд со статистикой и рейтингами из кэша Redis)
- **`/groups`** — Список всех тренировочных групп
- **`/groups/<group_id>`** — Состав конкретной группы
- **`/trainers`** — Список тренеров и их нагрузка
- **`/events`** — Список событий клуба (MongoDB)
- **`/manage`** — Управление данными (добавление тренеров, групп, участников и событий)
- **`/reviews`** — Отзывы на тренеров и клуб + средние оценки
- **`/analytics`** — Аналитика (SQL + MongoDB)
- **`/status`** — Статус доступности всех сервисов
- **`/api/status`** — JSON API для проверки статуса

### Формы
- Добавление тренера
- Добавление группы
- Добавление участника в группу
- Добавление события (MongoDB)
- Добавление отзыва на тренера
- Добавление отзыва на клуб

---

## Соответствие требованиям РГР

- ✅ Более 5 осмысленных страниц
- ✅ PostgreSQL: 5 таблиц в 3НФ, с JOIN и GROUP BY
- ✅ MongoDB: 3 коллекции + агрегация (`$group`, `$avg`)
- ✅ Redis: 4 ключа с TTL (120 секунд)
- ✅ Полная отказоустойчивость при падении любой БД
- ✅ Связь между PostgreSQL и MongoDB через `trainer_id`

---

## Запуск проекта

```bash
docker compose up --build
Приложение будет доступно по адресу:
http://localhost:8000
Запуск в фоновом режиме:
Bashdocker compose up --build -d

Управление сервисами
Bash# Остановить отдельный сервис
docker compose stop postgres
docker compose stop mongo
docker compose stop redis

# Запустить обратно
docker compose start postgres
docker compose start mongo
docker compose start redis

# Просмотр статуса контейнеров
docker compose ps

Структура проекта
text.
├── app/
│   ├── app.py                 # Основное приложение Flask
│   └── templates/             # HTML-шаблоны
├── scripts/
│   ├── init_postgres.sql      # Схема и тестовые данные PostgreSQL
│   └── init_mongo.js          # Тестовые данные MongoDB
├── docker-compose.yml
├── Dockerfile
├── REPORT.md                  # Отчёт по РГР
├── README.md
└── .env                       # (опционально) переменные окружения