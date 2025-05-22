import json, os, subprocess, time
from datetime import datetime

ALGORITHMS = [
    "DuelingDDQN",               # 1.1
    "QRDQN",                     # 1.2
    "DuelingDDQNPrioritized",    # 1.3
    "NoisyDuelingDDQN",          # 1.4
]

CONFIG_FILE = "config.json"
LOG_FILE = "run_all.log"

def log(message):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} — {message}\n")

def run_algorithm(name):
    print(f"\n🧠 Running {name}...\n")
    log(f"Started {name}")

    # Загружаем текущий конфиг
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Меняем алгоритм
    cfg["ALGORITHM"] = name
    cfg["METRICS_CSV"] = f"metrics_{name}.csv"
    cfg["VIDEO_PATH"] = f"video_{name}.mp4"

    # Сохраняем конфиг
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    start = time.time()
    subprocess.run(["python", "main.py"])
    duration = time.time() - start

    log(f"Finished {name} in {duration:.2f}s")
    log(f"→ CSV: {cfg['METRICS_CSV']}")
    log(f"→ MP4: {cfg['VIDEO_PATH']}")
    log("—" * 60)

if __name__ == "__main__":
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("🚀 Experiment Log — {}\n\n".format(datetime.now()))
    for algo in ALGORITHMS:
        run_algorithm(algo)
