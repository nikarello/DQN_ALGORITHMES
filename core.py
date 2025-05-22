
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Направления движения: up, down, left, right
DIRS = torch.tensor([[0, -1],
                     [0,  1],
                     [-1, 0],
                     [ 1, 0]], dtype=torch.long, device=device)

# Сдвиги для агентов 1x1, 2x2, 3x3
OFFS = {
    sz: torch.stack(
            torch.meshgrid(
                torch.arange(sz, device=device),
                torch.arange(sz, device=device),
                indexing="ij"
            ), dim=-1
        ).view(-1, 2)
    for sz in (1, 2, 3)
}

# Ядро для распространения огня
FIRE_KERNEL = torch.tensor(
    [[0,1,0],
     [1,0,1],
     [0,1,0]],
    device=device,
    dtype=torch.float32
)[None,None]

def batched_update_fire_exit(fire_mask, exit_mask, p_spread=0.5):
    inp    = fire_mask.unsqueeze(1)
    spread = F.conv2d(inp, FIRE_KERNEL, padding=1).squeeze(1)
    prob   = (spread>0) & (fire_mask==0) & (~exit_mask)
    newf   = (torch.rand_like(spread) < p_spread) & prob
    fire_mask = torch.clamp(fire_mask + newf.float(), max=1.0)
    return fire_mask, exit_mask

def batched_update_agents(positions, size, speed, knows_exit, alive, grid_size):
    B, N, _ = positions.shape

    # -- игнорируем мёртвых/эвакуированных --------------------------
    positions  = positions.clone()
    size       = size.clone()
    speed      = speed.clone()
    knows_exit = knows_exit.clone()

    positions[~alive]  = -1         # за пределы поля
    size     [~alive]  = 0
    speed    [~alive]  = 0
    knows_exit[~alive] = False
    
    H = W = grid_size
    ag_map  = torch.zeros((B, H, W), dtype=torch.float32, device=device)
    sz_map  = torch.zeros_like(ag_map)
    sp_map  = torch.zeros_like(ag_map)
    inf_map = torch.zeros_like(ag_map)

    size_cells = (size * 5).round().long()

    for sz in (1, 2, 3):
        mask = (size_cells == sz)  # [B, N]
        if not mask.any():
            continue

        b_idx, a_idx = torch.nonzero(mask, as_tuple=True)
        if b_idx.numel() == 0:
            continue

        pivots = positions[b_idx, a_idx]  # [M, 2]
        offs = OFFS[sz]                   # [K, 2]
        cells = pivots.unsqueeze(1) + offs.unsqueeze(0)  # [M, K, 2]
        cells = cells.view(-1, 2)
        xs, ys = cells[:, 0], cells[:, 1]

        K = sz * sz
        b_rep = b_idx.unsqueeze(1).expand(-1, K).reshape(-1)
        a_rep = a_idx.unsqueeze(1).expand(-1, K).reshape(-1)

        ag_map.index_put_((b_rep, ys, xs), torch.ones_like(xs, dtype=torch.float32), accumulate=False)
        sz_map.index_put_((b_rep, ys, xs), torch.full_like(xs, fill_value=sz/5.0, dtype=torch.float32), accumulate=False)

        sp_vals = speed[b_idx, a_idx]
        sp_rep = sp_vals.unsqueeze(1).expand(-1, K).reshape(-1)
        sp_map.index_put_((b_rep, ys, xs), sp_rep, accumulate=False)

        inf_vals = knows_exit[b_idx, a_idx].float()
        inf_rep = inf_vals.unsqueeze(1).expand(-1, K).reshape(-1)
        inf_map.index_put_((b_rep, ys, xs), inf_rep, accumulate=False)

    return ag_map, sz_map, sp_map, inf_map

def batched_get_views(ag_map, fire_mask, exit_mask, sz_map, sp_map, inf_map, positions, view_size):
    B, H, W = fire_mask.shape
    x = torch.stack([ag_map, fire_mask, exit_mask.float(), sz_map, sp_map, inf_map], dim=1)
    half = view_size // 2
    x_p = F.pad(x, (half, half, half, half))
    patches = F.unfold(x_p, kernel_size=view_size)
    B, Cx, L = patches.shape
    N = positions.shape[1]
    idx = positions[...,1] * W + positions[...,0]
    patches = patches.unsqueeze(1).expand(-1, N, -1, -1)
    idx_exp = idx.unsqueeze(2).expand(-1, -1, Cx)
    sel = torch.gather(patches, dim=3, index=idx_exp.unsqueeze(3))
    sel = sel.squeeze(3)
    views = sel.view(B, N, 6, view_size, view_size)
    return views

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.pos = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        import random
        idxs = random.sample(range(self.size), batch_size)
        batch = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states).to(device),
            torch.LongTensor(actions).unsqueeze(1).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.stack(next_states).to(device),
            torch.FloatTensor(dones).to(device)
        )

    def __len__(self):
        return self.size


def batched_step(positions, actions, size, speed, fire_mask, exit_mask, health):
    B, N, _ = positions.shape
    sp_cells = (speed * 4).round().long().unsqueeze(-1)
    delta    = DIRS[actions] * sp_cells

    # 1. Предложенные новые позиции
    prop = positions + delta
    prop = prop.clamp(min=0)
    maxc = (fire_mask.size(1) - (size * 5).round().long()).unsqueeze(-1)
    prop = torch.min(prop, maxc)

    # 2. Проверка коллизий
    grid_size = fire_mask.size(1)
    new_pos, coll_mask = _reject_collisions(prop, positions, size, speed, grid_size)

    # 3. Начисление базовых наград
    rewards = torch.full((B, N), -1.0, device=device)

    ys, xs = new_pos[..., 1], new_pos[..., 0]
    idx_flat = ys * fire_mask.size(2) + xs
    flat_fire = fire_mask.view(B, -1)
    hits = torch.gather(flat_fire, 1, idx_flat) > 0.5

    exs = torch.zeros((B, N), dtype=torch.bool, device=device)
    flat_exit = exit_mask.view(B, -1)

    for sz in (1, 2, 3):
        mask_sz = ((size * 5).round().long() == sz)
        if not mask_sz.any():
            continue
        b_idx, a_idx = torch.nonzero(mask_sz, as_tuple=True)
        pivots = new_pos[b_idx, a_idx]
        cells  = pivots.unsqueeze(1) + OFFS[sz].unsqueeze(0)
        cells  = cells.view(-1, 2)
        xs_c, ys_c = cells[:, 0], cells[:, 1]
        K = sz * sz
        b_rep = b_idx.unsqueeze(1).expand(-1, K).reshape(-1)
        a_rep = a_idx.unsqueeze(1).expand(-1, K).reshape(-1)
        ex_cells = exit_mask[b_rep, ys_c, xs_c]
        exs.index_put_((b_rep, a_rep), ex_cells.bool(), accumulate=True)

    # 4. Награды / Штрафы
    rewards = rewards + hits.float() * (-50.0) + exs.float() * 300.0
    health2 = health - hits.float() * 25.0
    died = hits & (health2 <= 0)
    rewards = rewards + died.float() * (-100.0)

    # 5. Обновление состояний
    health = health2.clamp(min=0.0)
    alive = ~(died | exs)
    f2, _ = batched_update_fire_exit(fire_mask, exit_mask)
    dones = ~alive.any(dim=1)

    return new_pos, rewards, dones, alive, health, f2, exit_mask, died, exs



# ─── helpers ──────────────────────────────────────────────────────────
def _reject_collisions(prop, positions, size, speed, grid_size):
    """Запретить ходы, если предложенная позиция пересекается с занятыми."""
    occ0, _, _, _ = batched_update_agents(
        positions, size, speed,
        knows_exit=torch.zeros_like(size, dtype=torch.bool, device=device),
        alive=torch.ones_like(size, dtype=torch.bool, device=device),
        grid_size=grid_size
    )

    B, N = size.shape
    coll = torch.zeros((B, N), dtype=torch.bool, device=device)

    for sz_val in (1, 2, 3):
        mask_sz = ((size * 5).round().long() == sz_val)
        if not mask_sz.any(): continue
        offs = OFFS[sz_val]
        b_idx, a_idx = torch.nonzero(mask_sz, as_tuple=True)
        pivots = prop[b_idx, a_idx]
        cells  = pivots.unsqueeze(1) + offs
        cells  = cells.view(-1, 2)
        xs, ys = cells[:, 0], cells[:, 1]
        K = sz_val * sz_val
        b_r = b_idx.unsqueeze(1).expand(-1, K).reshape(-1)
        a_r = a_idx.unsqueeze(1).expand(-1, K).reshape(-1)
        hit = occ0[b_r, ys, xs] > 0.5
        coll.index_put_((b_r, a_r), hit, accumulate=True)

    new_pos = torch.where(coll.unsqueeze(-1), positions, prop)
    return new_pos, coll
