# DQN_ALGORITHMES
Сравнение Dueling DDQN, QRDQN, Prioritized и Noisy DQN на задаче эвакуации агентов

# 🚒 Многоагентная Эвакуация с Deep Q-Learning

Этот проект моделирует эвакуацию агентов из здания с пожарами, используя современные алгоритмы глубокого обучения с подкреплением (RL). Поддерживаются следующие алгоритмы:

- ✅ Dueling Double DQN
- ✅ QR-DQN (Quantile Regression DQN)
- ✅ Dueling Double DQN с Prioritized Experience Replay (PER)
- ✅ Noisy Dueling Double DQN

---

## 📦 Состав проекта

```bash
full_jaga_jaga/
├── main.py                    # Основной запуск обучения
├── run_all.py                # Последовательный запуск всех алгоритмов
├── config.json               # Конфигурационный файл (используется run_all.py)
├── config_test_*.json        # Тестовые конфиги для отдельных запусков
├── algorithms/
│   ├── base_trainer.py       # Базовый класс обучения
│   ├── dueling_ddqn.py       # Классический Dueling DDQN
│   ├── qr_dqn.py             # Квантильный QR-DQN
│   ├── dueling_ddqn_prioritized.py  # Dueling DDQN + Prioritized Replay
│   └── noisy_dueling_ddqn.py        # Noisy-Net Dueling DDQN
├── core.py                   # Основная логика среды, движения, столкновений
├── environment.py            # Класс Environment
├── videos/                   # Сюда сохраняются видео эпизодов
├── metrics_*.csv             # Метрики обучения по каждому алгоритму
└── video_*.mp4               # Видео финальных эпизодов
```
🚀 Установка
1. Установите зависимости

pip install torch numpy matplotlib imageio pandas
🔥 Для сохранения видео требуется ffmpeg (обычно поставляется вместе с imageio)

⚙️ Запуск обучения
🧠 1. Один алгоритм
Для запуска одного алгоритма с заданным конфигом:


python main.py --config config_test_ddqn.json
Примеры конфигов:

config_test_ddqn.json — Dueling DDQN

config_test_qr.json — QR-DQN

config_test_ddqn_prioritized.json — Prioritized Dueling DDQN

config_test_noisy.json — Noisy Dueling DDQN

🔄 2. Все алгоритмы по очереди
Скрипт run_all.py последовательно запустит все 4 алгоритма:

python run_all.py
Результаты будут сохраняться в:

metrics_DuelingDDQN.csv, metrics_QRDQN.csv, ...

video_DuelingDDQN.mp4, video_QRDQN.mp4, ...

📊 Метрики
Во время обучения сохраняются метрики в CSV и строятся графики:

Reward — Суммарная награда эпизода

Evacuated / Died — Число эвакуированных и погибших

Avg steps to exit — Среднее число шагов до выхода

Avg HP at exit — Среднее здоровье у спасённых

Epsilon — Значение ε в ε-greedy стратегии

Loss — Средний loss по эпизоду

Пример графика:


Saved to → metrics_<algorithm>.csv
Saved to → video_<algorithm>.mp4
🎥 Видеоэпизоды
После обучения автоматически создаются видео (.mp4) с визуализацией 4 разных сред (если есть).

Агенты — синие

Огонь — красный

Выход — зелёный

Примеры:

video_DuelingDDQN.mp4

video_QRDQN.mp4

📌 Параметры в config.json

ALGORITHM	Название алгоритма (см. список выше)
NUM_EPISODES	Число эпизодов
NUM_ENVS	Сколько сред тренируются одновременно
BATCH_SIZE	Размер батча
LEARNING_RATE	Скорость обучения
MEMORY_SIZE	Размер replay buffer
AGENT_SPECS	Формат: [[кол-во, размер], ...] — размеры агентов
METRICS_CSV	Путь к файлу метрик
VIDEO_PATH	Путь к видеофайлу
