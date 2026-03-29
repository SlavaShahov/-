import json
import os
import socket
from datetime import UTC, datetime

from flask import Flask, jsonify, render_template, request
from psycopg import connect
from psycopg.rows import dict_row
from pymongo import MongoClient
from redis import Redis
from redis.exceptions import RedisError


app = Flask(__name__)


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def service_error_message(service: str) -> str:
    if service == "postgres":
        return "PostgreSQL сейчас недоступен. Проверьте, запущен ли контейнер postgres."
    if service == "mongo":
        return "MongoDB сейчас недоступен. Проверьте, запущен ли контейнер mongo."
    if service == "redis":
        return "Redis сейчас недоступен. Проверьте, запущен ли контейнер redis."
    return "Сервис сейчас недоступен."


PG_DSN = (
    f"host={_env('PG_HOST', 'localhost')} "
    f"port={_env('PG_PORT', '5432')} "
    f"dbname={_env('PG_DB', 'fight_club')} "
    f"user={_env('PG_USER', 'postgres')} "
    f"password={_env('PG_PASSWORD', 'postgres')} "
    "connect_timeout=2"
)

MONGO_URI = _env("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = _env("MONGO_DB", "fight_club")
REDIS_HOST = _env("REDIS_HOST", "localhost")
REDIS_PORT = int(_env("REDIS_PORT", "6379"))
REDIS_DB = int(_env("REDIS_DB", "0"))
REDIS_TTL_SECONDS = int(_env("REDIS_TTL_SECONDS", "120"))
INTERNET_CHECK_HOST = _env("INTERNET_CHECK_HOST", "example.com")


def check_internet() -> bool:
    try:
        socket.gethostbyname(INTERNET_CHECK_HOST)
        return True
    except OSError:
        return False


def pg_query(sql: str):
    try:
        with connect(PG_DSN, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                return cur.fetchall(), None
    except Exception:
        return [], service_error_message("postgres")


def pg_execute(sql: str, params: tuple):
    try:
        with connect(PG_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.commit()
        return None
    except Exception as e:
        err_str = str(e)
        if "unique" in err_str.lower() or "duplicate" in err_str.lower():
            return "DUPLICATE"
        return service_error_message("postgres")


def mongo_collection(name: str):
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1500)
        client.admin.command("ping")
        db = client[MONGO_DB]
        return db[name], None
    except Exception:
        return None, service_error_message("mongo")


def redis_client():
    try:
        r = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, socket_timeout=1.5)
        r.ping()
        return r, None
    except Exception:
        return None, service_error_message("redis")


def get_dashboard_stats():
    cache_key = "dashboard:stats"
    cached = None
    redis_error = None
    r, redis_error = redis_client()
    if r:
        try:
            raw = r.get(cache_key)
            if raw:
                cached = json.loads(raw)
        except RedisError as exc:
            redis_error = str(exc)

    if cached:
        return cached, redis_error, True

    rows, pg_error = pg_query(
        """
        SELECT
            (SELECT COUNT(*) FROM members) AS members_count,
            (SELECT COUNT(*) FROM groups) AS groups_count,
            (SELECT COUNT(*) FROM trainers) AS trainers_count
        """
    )
    if pg_error:
        return {"members_count": 0, "groups_count": 0, "trainers_count": 0}, pg_error, False

    data = rows[0] if rows else {"members_count": 0, "groups_count": 0, "trainers_count": 0}
    if r:
        try:
            r.set("dashboard:last_refresh", datetime.now(UTC).isoformat())
            r.setex(cache_key, REDIS_TTL_SECONDS, json.dumps(data, ensure_ascii=False))
        except RedisError as exc:
            redis_error = str(exc)
    return data, redis_error, False


def get_dashboard_ratings():
    club_key = "dashboard:ratings:club_avg"
    trainers_hash_key = "trainers:avg_ratings"
    redis_error = None
    r, redis_error = redis_client()
    cached_club = None
    cached_trainers = None

    if r:
        try:
            raw_club = r.get(club_key)
            raw_trainers = r.hgetall(trainers_hash_key)
            # hgetall возвращает пустой dict если ключа нет
            if raw_club and raw_trainers:
                cached_club = json.loads(raw_club)
                cached_trainers = [
                    json.loads(v) for v in raw_trainers.values()
                ]
        except RedisError as exc:
            redis_error = str(exc)

    if cached_club is not None and cached_trainers is not None:
        return {
            "club_avg_rating": cached_club,
            "trainers_avg": cached_trainers,
        }, redis_error, True

    feedback_col, feedback_error = mongo_collection("feedback")
    club_feedback_col, club_feedback_error = mongo_collection("club_feedback")
    mongo_error = feedback_error or club_feedback_error
    if mongo_error:
        return {"club_avg_rating": None, "trainers_avg": []}, mongo_error, False

    trainers_avg = list(
        feedback_col.aggregate(
            [
                # Группируем по trainer_id — связующее поле между MongoDB и PostgreSQL
                {"$group": {
                    "_id": "$trainer_id",
                    "trainer_name": {"$first": "$trainer_name"},
                    "avg_rating": {"$avg": "$rating"},
                    "count": {"$sum": 1}
                }},
                {"$sort": {"avg_rating": -1}},
            ]
        )
    )
    normalized_trainers = [
        {
            "trainer_id": row["_id"],
            "trainer_name": row["trainer_name"],
            "avg_rating": float(row["avg_rating"]),
            "count": int(row["count"]),
        }
        for row in trainers_avg
    ]

    club_avg_rows = list(club_feedback_col.aggregate([{"$group": {"_id": None, "avg_rating": {"$avg": "$rating"}}}]))
    club_avg_rating = float(club_avg_rows[0]["avg_rating"]) if club_avg_rows else None

    if r:
        try:
            r.setex(club_key, REDIS_TTL_SECONDS, json.dumps(club_avg_rating, ensure_ascii=False))
            # Сохраняем Hash: поле = trainer_id, значение = JSON с данными тренера
            # Так Redis явно связан с PostgreSQL и MongoDB через trainer_id
            for row in normalized_trainers:
                r.hset(trainers_hash_key, str(row["trainer_id"]), json.dumps(row, ensure_ascii=False))
            r.expire(trainers_hash_key, REDIS_TTL_SECONDS)
        except RedisError as exc:
            redis_error = str(exc)

    return {
        "club_avg_rating": club_avg_rating,
        "trainers_avg": normalized_trainers,
    }, redis_error, False


@app.route("/")
def dashboard():
    stats, data_error, from_cache = get_dashboard_stats()
    ratings, ratings_error, ratings_from_cache = get_dashboard_ratings()
    return render_template(
        "dashboard.html",
        title="Главная",
        stats=stats,
        from_cache=from_cache,
        data_error=data_error,
        ratings=ratings,
        ratings_error=ratings_error,
        ratings_from_cache=ratings_from_cache,
    )


@app.route("/groups")
def groups_page():
    rows, error = pg_query(
        """
        SELECT
            g.group_id,
            g.name AS group_name,
            ma.name AS martial_art,
            COALESCE(t.full_name, 'Не назначен') AS trainer_name,
            COUNT(s.member_id) AS members_count
        FROM groups g
        JOIN martialarts ma ON ma.art_id = g.art_id
        LEFT JOIN trainers t ON t.trainer_id = g.trainer_id
        LEFT JOIN subscriptions s ON s.group_id = g.group_id
        GROUP BY g.group_id, g.name, ma.name, t.full_name
        ORDER BY members_count DESC, g.name;
        """
    )
    return render_template("groups.html", title="Группы", rows=rows, error=error)


@app.route("/groups/<int:group_id>")
def group_detail_page(group_id: int):
    group_rows, group_error = pg_query(
        f"""
        SELECT
            g.group_id,
            g.name AS group_name,
            ma.name AS martial_art,
            COALESCE(t.full_name, 'Не назначен') AS trainer_name,
            g.level
        FROM groups g
        JOIN martialarts ma ON ma.art_id = g.art_id
        LEFT JOIN trainers t ON t.trainer_id = g.trainer_id
        WHERE g.group_id = {group_id}
        LIMIT 1;
        """
    )
    members_rows, members_error = pg_query(
        f"""
        SELECT
            m.member_id,
            m.full_name,
            m.birth_date,
            m.gender,
            s.start_date,
            s.end_date
        FROM subscriptions s
        JOIN members m ON m.member_id = s.member_id
        WHERE s.group_id = {group_id}
        ORDER BY m.full_name;
        """
    )
    error = group_error or members_error
    group_info = group_rows[0] if group_rows else None
    return render_template(
        "group_detail.html",
        title="Состав группы",
        group_info=group_info,
        members_rows=members_rows,
        error=error,
    )


@app.route("/trainers")
def trainers_page():
    rows, error = pg_query(
        """
        SELECT
            t.full_name AS trainer_name,
            t.experience_years,
            COALESCE(COUNT(g.group_id), 0) AS groups_count
        FROM trainers t
        LEFT JOIN groups g ON g.trainer_id = t.trainer_id
        GROUP BY t.trainer_id, t.full_name, t.experience_years
        ORDER BY t.full_name;
        """
    )
    return render_template("trainers.html", title="Тренеры", rows=rows, error=error)


@app.route("/manage", methods=["GET", "POST"])
def manage_page():
    message = None
    error = None
    postgres_ok = True
    mongo_ok = True

    # Получаем данные для форм заранее — они нужны и для GET и внутри POST-блоков
    arts, arts_error = pg_query("SELECT art_id, name FROM martialarts ORDER BY name")
    trainers, trainers_error = pg_query("SELECT trainer_id, full_name FROM trainers ORDER BY full_name")
    groups, groups_error = pg_query("SELECT group_id, name FROM groups ORDER BY name")

    if request.method == "POST":
        action = request.form.get("action", "")

        # ========== ДОБАВЛЕНИЕ ТРЕНЕРА ==========
        if action == "add_trainer":
            full_name = request.form.get("full_name", "").strip()
            phone = request.form.get("phone", "").strip()
            experience_years = request.form.get("experience_years", "").strip()
            if not full_name or not phone or not experience_years:
                error = "Заполните все поля тренера."
            else:
                err = pg_execute(
                    "INSERT INTO trainers (full_name, experience_years, phone) VALUES (%s, %s, %s)",
                    (full_name, int(experience_years), phone),
                )
                if err == "DUPLICATE":
                    error = f"Тренер с телефоном {phone} уже существует."
                elif err:
                    error = err
                    postgres_ok = False
                else:
                    message = f"✅ Тренер '{full_name}' успешно добавлен!"

        # ========== ДОБАВЛЕНИЕ ГРУППЫ ==========
        elif action == "add_group":
            name = request.form.get("name", "").strip()
            art_id = request.form.get("art_id", "").strip()
            trainer_id = request.form.get("trainer_id", "").strip()
            level = request.form.get("level", "").strip()

            # Получаем название вида спорта для сообщения
            art_name = ""
            for art in arts:
                if str(art["art_id"]) == art_id:
                    art_name = art["name"]
                    break

            if not name or not art_id or not level:
                error = "Заполните обязательные поля группы."
            else:
                trainer_value = int(trainer_id) if trainer_id else None
                # Проверяем существование группы с таким же названием
                existing, _ = pg_query(f"SELECT group_id FROM groups WHERE name = '{name}' LIMIT 1")
                if existing:
                    error = f"Группа с названием '{name}' уже существует."
                else:
                    err = pg_execute(
                        "INSERT INTO groups (name, art_id, trainer_id, level) VALUES (%s, %s, %s, %s)",
                        (name, int(art_id), trainer_value, level),
                    )
                    if err:
                        error = err
                        postgres_ok = False
                    else:
                        message = f"✅ Группа '{name}' ({art_name}, {level}) успешно добавлена!"

        # ========== ДОБАВЛЕНИЕ УЧАСТНИКА В ГРУППУ ==========
        elif action == "add_member_to_group":
            full_name = request.form.get("member_full_name", "").strip()
            birth_date = request.form.get("birth_date", "").strip()
            phone = request.form.get("member_phone", "").strip()
            gender = request.form.get("gender", "").strip()
            group_id = request.form.get("group_id", "").strip()
            start_date = request.form.get("start_date", "").strip()
            end_date = request.form.get("end_date", "").strip()
            price = request.form.get("price", "").strip()

            # Получаем название группы для сообщения
            group_name = ""
            for group in groups:
                if str(group["group_id"]) == group_id:
                    group_name = group["name"]
                    break

            if not all([full_name, birth_date, gender, group_id, start_date, end_date, price]):
                error = "Заполните все обязательные поля участника."
            else:
                try:
                    with connect(PG_DSN, row_factory=dict_row) as conn:
                        with conn.cursor() as cur:
                            # Проверяем — есть ли уже участник с таким телефоном
                            if phone:
                                cur.execute("SELECT member_id FROM members WHERE phone = %s LIMIT 1", (phone,))
                                existing_member = cur.fetchone()
                            else:
                                existing_member = None

                            if existing_member:
                                # Участник уже есть — проверяем не записан ли он уже в эту группу
                                member_id = existing_member["member_id"]
                                cur.execute(
                                    "SELECT subscription_id FROM subscriptions WHERE member_id = %s AND group_id = %s LIMIT 1",
                                    (member_id, int(group_id))
                                )
                                if cur.fetchone():
                                    error = f"Участник с телефоном {phone} уже записан в эту группу."
                                else:
                                    cur.execute(
                                        """INSERT INTO subscriptions
                                        (member_id, group_id, start_date, end_date, price)
                                        VALUES (%s, %s, %s, %s, %s)""",
                                        (member_id, int(group_id), start_date, end_date, float(price)),
                                    )
                                    conn.commit()
                                    message = f"✅ Участник '{full_name}' уже существовал — добавлен в группу '{group_name}' по новому абонементу!"
                            else:
                                # Новый участник — создаём и сразу записываем
                                cur.execute(
                                    """INSERT INTO members (full_name, birth_date, phone, gender)
                                    VALUES (%s, %s, %s, %s) RETURNING member_id""",
                                    (full_name, birth_date, phone or None, gender),
                                )
                                member_id = cur.fetchone()["member_id"]
                                cur.execute(
                                    """INSERT INTO subscriptions
                                    (member_id, group_id, start_date, end_date, price)
                                    VALUES (%s, %s, %s, %s, %s)""",
                                    (member_id, int(group_id), start_date, end_date, float(price)),
                                )
                                conn.commit()
                                message = f"✅ Участник '{full_name}' успешно добавлен и записан в группу '{group_name}'!"
                except Exception as e:
                    if not error:
                        error = service_error_message("postgres")
                        postgres_ok = False

        # ========== ДОБАВЛЕНИЕ СОБЫТИЯ (MongoDB) ==========
        elif action == "add_event":
            events_col, mongo_error = mongo_collection("events")
            if mongo_error:
                error = mongo_error
                mongo_ok = False
            else:
                title = request.form.get("title", "").strip()
                martial_art = request.form.get("martial_art", "").strip()
                event_date = request.form.get("event_date", "").strip()
                participants_limit = request.form.get("participants_limit", "").strip()
                if not title or not martial_art or not event_date or not participants_limit:
                    error = "Заполните все поля события."
                else:
                    try:
                        event_dt = datetime.fromisoformat(event_date)
                        # Проверяем — нет ли уже события с таким названием и датой
                        existing = events_col.find_one({"title": title, "event_date": event_dt})
                        if existing:
                            error = f"Событие '{title}' на эту дату уже существует."
                        else:
                            events_col.insert_one(
                                {
                                    "title": title,
                                    "martial_art": martial_art,
                                    "event_date": event_dt,
                                    "participants_limit": int(participants_limit),
                                }
                            )
                            message = f"✅ Событие '{title}' ({martial_art}) успешно добавлено в MongoDB!"
                    except Exception:
                        error = service_error_message("mongo")
                        mongo_ok = False

    if arts_error or trainers_error or groups_error:
        postgres_ok = False
    _, mongo_error = mongo_collection("events")
    if mongo_error:
        mongo_ok = False
    error = error or arts_error or trainers_error or groups_error

    return render_template(
        "manage.html",
        title="Управление",
        arts=arts,
        trainers=trainers,
        groups=groups,
        error=error,
        message=message,
        postgres_ok=postgres_ok,
        mongo_ok=mongo_ok,
    )


@app.route("/events")
def events_page():
    events_col, error = mongo_collection("events")
    rows = []
    if events_col is not None:
        rows = list(events_col.find({}, {"_id": 0}).sort("event_date", 1))
    return render_template("events.html", title="События", rows=rows, error=error)


@app.route("/reviews", methods=["GET", "POST"])
def reviews_page():
    message = None
    error = None

    trainer_reviews_col, trainer_col_error = mongo_collection("feedback")
    club_reviews_col, club_col_error = mongo_collection("club_feedback")
    error = trainer_col_error or club_col_error

    if request.method == "POST" and not error:
        action = request.form.get("action", "")
        try:
            rating = int(request.form.get("rating", "0"))
        except ValueError:
            rating = 0
        if rating < 1 or rating > 5:
            error = "Оценка должна быть от 1 до 5."
        elif action == "add_trainer_review":
            member_name = request.form.get("member_name", "").strip()
            trainer_name = request.form.get("trainer_name", "").strip()
            comment = request.form.get("comment", "").strip()
            if not member_name or not trainer_name:
                error = "Укажите автора и тренера."
            else:
                # Получаем trainer_id из PostgreSQL — связующее поле между всеми тремя БД
                id_rows, _ = pg_query(
                    f"SELECT trainer_id FROM trainers WHERE full_name = '{trainer_name}' LIMIT 1"
                )
                trainer_id = id_rows[0]["trainer_id"] if id_rows else None
                trainer_reviews_col.insert_one(
                    {
                        "trainer_id": trainer_id,
                        "member_name": member_name,
                        "trainer_name": trainer_name,
                        "rating": rating,
                        "comment": comment,
                        "created_at": datetime.now(UTC),
                    }
                )
                message = "Отзыв на тренера добавлен."
        elif action == "add_club_review":
            author = request.form.get("author", "").strip()
            comment = request.form.get("comment", "").strip()
            if not author:
                error = "Укажите автора отзыва на клуб."
            else:
                club_reviews_col.insert_one(
                    {
                        "author": author,
                        "rating": rating,
                        "comment": comment,
                        "created_at": datetime.now(UTC),
                    }
                )
                message = "Отзыв на клуб добавлен."

    trainer_reviews = []
    club_reviews = []
    trainer_avg_rows = []
    club_avg_rating = None

    if not error:
        trainer_reviews = list(trainer_reviews_col.find({}).sort("created_at", -1).limit(20))
        club_reviews = list(club_reviews_col.find({}).sort("created_at", -1).limit(20))
        trainer_avg_rows = list(
            trainer_reviews_col.aggregate(
                [
                    {"$group": {
                        "_id": "$trainer_id",
                        "trainer_name": {"$first": "$trainer_name"},
                        "avg_rating": {"$avg": "$rating"},
                        "count": {"$sum": 1}
                    }},
                    {"$sort": {"avg_rating": -1}},
                ]
            )
        )
        club_avg = list(
            club_reviews_col.aggregate([{"$group": {"_id": None, "avg_rating": {"$avg": "$rating"}}}])
        )
        if club_avg:
            club_avg_rating = float(club_avg[0]["avg_rating"])

    trainers, trainers_error = pg_query("SELECT full_name FROM trainers ORDER BY full_name")
    error = error or trainers_error

    return render_template(
        "reviews.html",
        title="Отзывы",
        trainers=trainers,
        trainer_reviews=trainer_reviews,
        club_reviews=club_reviews,
        trainer_avg_rows=trainer_avg_rows,
        club_avg_rating=club_avg_rating,
        error=error,
        message=message,
    )


@app.route("/analytics")
def analytics_page():
    pg_rows, pg_error = pg_query(
        """
        SELECT
            ma.name AS martial_art,
            ROUND(AVG(s.price), 2) AS avg_subscription_price
        FROM subscriptions s
        JOIN groups g ON s.group_id = g.group_id
        JOIN martialarts ma ON g.art_id = ma.art_id
        GROUP BY ma.name
        ORDER BY avg_subscription_price DESC;
        """
    )

    feedback_col, mongo_error = mongo_collection("feedback")
    club_feedback_col, club_mongo_error = mongo_collection("club_feedback")
    mongo_rows = []
    club_avg_rating = None
    if feedback_col is not None:
        pipeline = [
            {"$group": {
                "_id": "$trainer_id",
                "trainer_name": {"$first": "$trainer_name"},
                "avg_rating": {"$avg": "$rating"},
                "count": {"$sum": 1}
            }},
            {"$sort": {"avg_rating": -1}},
        ]
        mongo_rows = list(feedback_col.aggregate(pipeline))

    if club_feedback_col is not None:
        avg_rows = list(club_feedback_col.aggregate([{"$group": {"_id": None, "avg_rating": {"$avg": "$rating"}}}]))
        if avg_rows:
            club_avg_rating = float(avg_rows[0]["avg_rating"])

    combined_mongo_error = mongo_error or club_mongo_error
    return render_template(
        "analytics.html",
        title="Аналитика",
        pg_rows=pg_rows,
        pg_error=pg_error,
        mongo_rows=mongo_rows,
        mongo_error=combined_mongo_error,
        club_avg_rating=club_avg_rating,
    )


@app.route("/status")
def status_page():
    _, pg_error = pg_query("SELECT 1 AS ok")
    _, mongo_error = mongo_collection("events")
    _, redis_error = redis_client()
    online = check_internet()
    status = {
        "internet": {"ok": online, "error": None if online else "Нет доступа к сети"},
        "postgres": {"ok": pg_error is None, "error": pg_error},
        "mongo": {"ok": mongo_error is None, "error": mongo_error},
        "redis": {"ok": redis_error is None, "error": redis_error},
    }
    return render_template("status.html", title="Статус сервисов", status=status)


@app.route("/api/status")
def api_status():
    _, pg_error = pg_query("SELECT 1 AS ok")
    _, mongo_error = mongo_collection("events")
    _, redis_error = redis_client()
    return jsonify(
        {
            "internet": check_internet(),
            "postgres": pg_error is None,
            "mongo": mongo_error is None,
            "redis": redis_error is None,
        }
    )


if __name__ == "__main__":
    app.run(
        host=_env("FLASK_HOST", "0.0.0.0"),
        port=int(_env("FLASK_PORT", "8000")),
        debug=False,
    )