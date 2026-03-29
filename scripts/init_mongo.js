db = db.getSiblingDB("fight_club");

db.events.insertMany([
  {
    title: "Открытый спарринг по боксу",
    martial_art: "Бокс",
    event_date: new Date("2026-04-10T18:00:00Z"),
    participants_limit: 30
  },
  {
    title: "Семинар по каратэ",
    martial_art: "Каратэ",
    event_date: new Date("2026-04-15T16:00:00Z"),
    participants_limit: 25
  },
  {
    title: "Турнир по дзюдо",
    martial_art: "Дзюдо",
    event_date: new Date("2026-04-20T11:00:00Z"),
    participants_limit: 40
  }
]);

db.feedback.insertMany([
  {
    trainer_id: 1,
    member_name: "Кузнецов Андрей",
    trainer_name: "Иванов Сергей",
    rating: 5,
    comment: "Отличные тренировки",
    created_at: new Date("2026-03-10T10:00:00Z")
  },
  {
    trainer_id: 2,
    member_name: "Смирнова Ольга",
    trainer_name: "Петров Алексей",
    rating: 4,
    comment: "Насыщенная программа",
    created_at: new Date("2026-03-12T10:00:00Z")
  },
  {
    trainer_id: 3,
    member_name: "Иванова Мария",
    trainer_name: "Сидорова Анна",
    rating: 5,
    comment: "Очень доступные объяснения",
    created_at: new Date("2026-03-20T10:00:00Z")
  }
]);

db.club_feedback.insertMany([
  {
    author: "Гость",
    rating: 5,
    comment: "Отличная атмосфера и сильные тренеры",
    created_at: new Date("2026-03-22T10:00:00Z")
  },
  {
    author: "Участник клуба",
    rating: 4,
    comment: "Хорошее расписание, но хотелось бы больше вечерних групп",
    created_at: new Date("2026-03-25T10:00:00Z")
  }
]);