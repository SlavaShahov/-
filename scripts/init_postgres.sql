CREATE TABLE IF NOT EXISTS MartialArts (
    art_id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    description TEXT
);

CREATE TABLE IF NOT EXISTS Trainers (
    trainer_id SERIAL PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    experience_years INT CHECK (experience_years >= 1),
    phone VARCHAR(20) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS Groups (
    group_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    art_id INT NOT NULL REFERENCES MartialArts(art_id) ON DELETE CASCADE,
    trainer_id INT REFERENCES Trainers(trainer_id) ON DELETE SET NULL,
    level VARCHAR(20) CHECK (level IN ('начинающий', 'средний', 'продвинутый'))
);

CREATE TABLE IF NOT EXISTS Members (
    member_id SERIAL PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    birth_date DATE NOT NULL,
    phone VARCHAR(20) UNIQUE,
    gender CHAR(1) CHECK (gender IN ('M','F')),
    CHECK (birth_date <= CURRENT_DATE - INTERVAL '5 year')
);

CREATE TABLE IF NOT EXISTS Subscriptions (
    subscription_id SERIAL PRIMARY KEY,
    member_id INT NOT NULL REFERENCES Members(member_id) ON DELETE CASCADE,
    group_id INT NOT NULL REFERENCES Groups(group_id) ON DELETE CASCADE,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    price NUMERIC(8,2) CHECK (price > 0),
    CHECK (end_date > start_date)
);

CREATE TABLE IF NOT EXISTS Attendances (
    attendance_id SERIAL PRIMARY KEY,
    subscription_id INT NOT NULL REFERENCES Subscriptions(subscription_id) ON DELETE CASCADE,
    attendance_date DATE NOT NULL
);

INSERT INTO MartialArts (name, description) VALUES
('Бокс', 'Удары руками и защита'),
('Каратэ', 'Японское боевое искусство'),
('Дзюдо', 'Броски и удержания'),
('Рукопашный бой', 'Боевые приёмы без оружия')
ON CONFLICT (name) DO NOTHING;

INSERT INTO Trainers (full_name, experience_years, phone) VALUES
('Иванов Сергей', 8, '+79990001111'),
('Петров Алексей', 5, '+79990002222'),
('Сидорова Анна', 10, '+79990003333'),
('Кузьмин Олег', 12, '+79990004444')
ON CONFLICT (phone) DO NOTHING;

INSERT INTO Groups (name, art_id, trainer_id, level) VALUES
('Бокс для начинающих', 1, 1, 'начинающий'),
('Каратэ продвинутая', 2, 2, 'продвинутый'),
('Дзюдо средняя', 3, 3, 'средний'),
('Рукопашный бой начинающий', 4, 4, 'начинающий'),
('Рукопашный бой продвинутый', 4, 4, 'продвинутый')
ON CONFLICT DO NOTHING;

INSERT INTO Members (full_name, birth_date, phone, gender) VALUES
('Кузнецов Андрей', '2003-06-12', '+79995551111', 'M'),
('Смирнова Ольга', '2005-02-21', '+79995552222', 'F'),
('Орлов Михаил', '2008-09-10', '+79995553333', 'M'),
('Иванова Мария', '2006-03-15', '+79995554444', 'F'),
('Лебедев Артём', '2002-11-01', '+79995555555', 'M'),
('Фёдорова Елена', '2007-04-09', '+79995556666', 'F')
ON CONFLICT (phone) DO NOTHING;

INSERT INTO Subscriptions (member_id, group_id, start_date, end_date, price) VALUES
(1, 1, '2026-03-01', '2026-03-31', 2500),
(2, 2, '2026-03-01', '2026-04-01', 3000),
(3, 1, '2026-03-05', '2026-04-05', 2500),
(4, 3, '2026-03-10', '2026-04-10', 2700),
(5, 4, '2026-03-01', '2026-03-31', 2800),
(6, 5, '2026-03-03', '2026-04-03', 3200)
ON CONFLICT DO NOTHING;

INSERT INTO Attendances (subscription_id, attendance_date) VALUES
(1, '2026-03-05'), (1, '2026-03-12'),
(2, '2026-03-06'), (2, '2026-03-13'),
(3, '2026-03-07'), (3, '2026-03-14'),
(4, '2026-03-08'), (4, '2026-03-15')
ON CONFLICT DO NOTHING;
