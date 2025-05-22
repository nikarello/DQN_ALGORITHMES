# algorithms/base_trainer.py
import abc, time, torch

import csv, pathlib, pandas as pd, matplotlib.pyplot as plt
import math, numpy as np

import imageio, os
from PIL import Image
import numpy as np
from datetime import datetime

from environment import stack_envs
from core import batched_update_fire_exit, batched_update_agents, \
                batched_get_views, batched_step

class BaseTrainer(abc.ABC):
    def __init__(self, envs, cfg):
        self.envs=envs; self.cfg=cfg
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online = None
        self.target = None
        # 🔧 Универсальные параметры
        self.in_ch      = cfg["INPUT_CHANNELS"]
        self.view       = cfg["VIEW_SIZE"]
        self.lr         = cfg["LEARNING_RATE"]
        self.gamma      = cfg["GAMMA"]
        self.batch_size = cfg["BATCH_SIZE"]
        self.memory_size= cfg.get("MEMORY_SIZE", 100_000)
        self.eps_start  = cfg.get("EPSILON_START", 1.0)
        self.eps_min    = cfg.get("EPSILON_MIN", 0.1)
        self.eps_decay  = cfg.get("EPSILON_DECAY", 0.995)
        self.n_envs     = cfg["NUM_ENVS"]
        self.max_steps  = cfg["MAX_STEPS_PER_EPISODE"]
        self.target_update_freq = cfg.get("TARGET_UPDATE_FREQ", 5)
        self.grid_size  = cfg["GRID_SIZE"]
        self.agent_specs= cfg["AGENT_SPECS"]

    # ---- абстрактные ----
    @abc.abstractmethod
    def build_model(self): ...
    @abc.abstractmethod
    def select_actions(self, views, mask): ...
    @abc.abstractmethod
    def learn_step(self): ...
    @abc.abstractmethod
    def after_episode(self, ep): ...
    # ---- общий цикл ----

    
    def train(self):

        B = len(self.envs)
        for e in self.envs:
            e.reset()

        self._init_metrics()
        eps = getattr(self, "epsilon", float("nan"))

        for ep in range(self.cfg["NUM_EPISODES"]):
            start = time.time()
            pos, alive, know, hp, sz, sp, fire, exit_m = stack_envs(self.envs)
            done = torch.zeros(B, dtype=torch.bool, device=self.device)

            exited_mask = torch.zeros((B, self.num_agents), dtype=torch.bool, device=self.device)
            died_mask   = torch.zeros_like(exited_mask)
            exit_step   = torch.full_like(exited_mask, -1, dtype=torch.int32)
            hp_at_exit  = torch.zeros_like(exited_mask, dtype=torch.float)

            total_reward = 0.0

            for step in range(1, self.max_steps + 1):
                fire, exit_m = batched_update_fire_exit(fire, exit_m)
                ag, szm, spm, inf = batched_update_agents(pos, sz, sp, know, alive, self.grid_size)
                views = batched_get_views(ag, fire, exit_m, szm, spm, inf, pos, self.view)

                mask = alive & (~know)
                actions = self.select_actions(views, mask)

                next_pos, rewards, dones, alive, hp, fire, exit_m, died, exits = \
                    batched_step(pos, actions, sz, sp, fire, exit_m, hp)

                next_ag, next_szm, next_spm, next_inf = \
                    batched_update_agents(next_pos, sz, sp, know, alive, self.grid_size)
                next_views = batched_get_views(
                    next_ag, fire, exit_m, next_szm, next_spm, next_inf, next_pos, self.view
                )

                for b in range(B):
                    idxs = torch.nonzero(mask[b], as_tuple=False).squeeze(1)
                    for i in idxs.tolist():
                        self.store_transition(
                            views[b, i].detach().cpu(),
                            actions[b, i].item(),
                            rewards[b, i].item(),
                            next_views[b, i].detach().cpu(),
                            dones[b].item()
                        )

                # метрики
                total_reward += rewards.sum().item()
                newly_exit = exits & (~exited_mask)
                newly_die  = died  & (~died_mask)
                exited_mask |= newly_exit
                died_mask   |= newly_die
                exit_step[newly_exit]  = step
                hp_at_exit[newly_exit] = hp[newly_exit]

                pos = next_pos
                done |= dones

                l = self.learn_step()
                if l is not None and l > 0:
                    self.loss_sum += l
                    self.loss_cnt += 1

                if done.all():
                    break

            evac_cnt = int(exited_mask.sum().item())
            died_cnt = int(died_mask.sum().item())
            avg_st   = float(exit_step[exited_mask].float().mean().item()) if evac_cnt else math.nan
            avg_hp   = float(hp_at_exit[exited_mask].mean().item())         if evac_cnt else math.nan
            avg_loss = self.loss_sum / self.loss_cnt if self.loss_cnt else math.nan

            self.metrics.append({
                "episode":   ep + 1,
                "reward":    float(total_reward),
                "evacuated": evac_cnt,
                "died":      died_cnt,
                "avg_steps": avg_st,
                "avg_hp":    avg_hp,
                "epsilon":   float(getattr(self, "epsilon", float("nan"))),
                "loss":      avg_loss,
                "duration":  time.time() - start
            })

            self.loss_sum, self.loss_cnt = 0.0, 0
            self.after_episode(ep)
            print(f"EP {ep + 1} finished in {time.time() - start:.2f}s")

        self._save_metrics_csv()
        self._plot_metrics()

        for i in range(4):
            if i >= len(self.envs): break 
            self._render_episode(env_idx=i)


    def store_transition(self, s, a, r, ns, d):
        if hasattr(self, "replay"):
            self.replay.push(s, a, r, ns, d)


    def _init_metrics(self):
        self.metrics = []                       
        self.loss_sum, self.loss_cnt = 0.0, 0
        # для эвакуации / смертей
        env0 = self.envs[0]
        self.num_agents = env0.N              

    def _save_metrics_csv(self):
        fname = self.cfg.get("METRICS_CSV", "metrics.csv")
        df = pd.DataFrame(self.metrics)
        out = pathlib.Path(fname)
        df.to_csv(out, index=False)
        print(f"[√] metrics saved → {out.resolve()}")

    def _plot_metrics(self):
        df = pd.DataFrame(self.metrics)
        plt.figure(figsize=(12, 10))

        algo = self.cfg.get("ALGORITHM", "Algorithm")
        plt.suptitle(f"{algo} — Training Metrics", fontsize=16)

        plt.subplot(3, 2, 1); plt.plot(df["episode"], df["reward"]); plt.title("Reward")

        plt.subplot(3, 2, 2)
        plt.plot(df["episode"], df["evacuated"], label="evacuated")
        plt.plot(df["episode"], df["died"], label="died")
        plt.legend(); plt.title("Evacuated / Died")

        plt.subplot(3, 2, 3); plt.plot(df["episode"], df["avg_steps"]); plt.title("Avg steps to exit")
        plt.subplot(3, 2, 4); plt.plot(df["episode"], df["avg_hp"]); plt.title("Avg HP at exit")
        plt.subplot(3, 2, 5); plt.plot(df["episode"], df["epsilon"]); plt.title("Epsilon")
        plt.subplot(3, 2, 6); plt.plot(df["episode"], df["loss"]);    plt.title("Loss")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


    # -----------------------------------------------------------------------
    # UNIVERSAL RENDER ------------------------------------------------------
    # -----------------------------------------------------------------------
    def _render_episode(
        self,
        env_idx: int = 0,
        max_steps: int | None = None,
        out_dir: str = "videos",
        scale: int = 10,
    ):
        """Проиграть один эпизод без ε-шума и записать MP4-ролик.

        • env_idx   – какую среду взять (0-based)  
        • max_steps – обрезка по шагам (None → self.max_steps)  
        • scale     – во сколько раз «увеличить пиксели» для красивой картинки
        """
        os.makedirs(out_dir, exist_ok=True)
        env       = self.envs[env_idx]
        algo_name = self.__class__.__name__.replace("Trainer", "")
        fname     = os.path.join(
            out_dir,
            f"{algo_name}_env{env_idx+1:02d}.mp4"
        )

        # ---------- подготовка сети к дет-режиму -------------
        prev_eps          = getattr(self, "epsilon", None)
        self.epsilon      = 0.0                # полностью greedy
        noisy             = hasattr(self.online, "reset_noise")
        self.online.eval()
        self.target.eval()
        if noisy:                                 # фиксируем шум один раз
            self.online.reset_noise()
            self.target.reset_noise()

        env.reset()
        grid = self.grid_size
        frames = []

        max_steps = max_steps or self.max_steps
        pos, alive, know, hp, sz, sp, fire, exit_m = stack_envs([env])

        for _ in range(max_steps):
            ag, szm, spm, inf = batched_update_agents(
                pos, sz, sp, know, alive, self.grid_size
            )
            views = batched_get_views(
                ag, fire, exit_m, szm, spm, inf, pos, self.view
            )
            actions = self.select_actions(views, alive & (~know))

            pos, _, done, alive, hp, fire, exit_m, _, _ = batched_step(
                pos, actions, sz, sp, fire, exit_m, hp
            )

            # --- рисуем -----------------------------------------------------------------
            rgb = np.zeros((grid, grid, 3), dtype=np.uint8)
            rgb[exit_m[0].cpu().numpy() == 1] = (0, 255, 0)          # выход – зелёный
            rgb[fire  [0].cpu().numpy() == 1] = (255, 0, 0)          # огонь – красный
            ys, xs = np.where(ag[0].cpu().numpy() > 0.5)             # агенты – синий
            rgb[ys, xs] = (0, 0, 255)

            pil = Image.fromarray(rgb).resize(
                (grid * scale, grid * scale), Image.NEAREST
            )
            frames.append(np.asarray(pil))

            if done.item():
                break
            # распространение огня – ОБЯЗАТЕЛЬНО в конце цикла,
            # иначе последний кадр не совпадёт с «шагом среды»
            fire, exit_m = batched_update_fire_exit(fire, exit_m)

        # ---------- запись video ----------------------------------------------------------
        with imageio.get_writer(
            fname, fps=5, codec="libx264",
            ffmpeg_params=["-pix_fmt", "yuv420p", "-profile:v", "baseline"]
        ) as vid:
            for f in frames:
                vid.append_data(f)

        # ---------- восстановление состояний ---------------------------------------------
        if prev_eps is not None:
            self.epsilon = prev_eps
        print(f"[√] video saved → {fname!s}")