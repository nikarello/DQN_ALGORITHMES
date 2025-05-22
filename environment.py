# environment.py
import random, torch
from core import device, OFFS, DIRS, batched_update_fire_exit

class Agent:
    def __init__(self, idx, grid, size, speed, occupied):
        self.id=idx; self.size=size; self.speed=speed; self.grid=grid
        self.position=self._random_pos(occupied); self.health=100
        self.alive=True; self.exited=False; self.knows_exit=(idx==0)
    def _random_pos(self, occ):
        while True:
            x=random.randint(0,self.grid-self.size)
            y=random.randint(0,self.grid-self.size)
            cells={(x+i,y+j) for i in range(self.size) for j in range(self.size)}
            if not (cells&occ): return (x,y)
    def cells(self):
        x,y=self.position
        return {(x+i,y+j) for i in range(self.size) for j in range(self.size)}

class Environment:
    def __init__(self, grid, agent_specs, num_fires, exit_pos):
        self.grid=grid; self.exit_positions=exit_pos; self.num_fires=num_fires
        self.exit_mask=torch.zeros((grid,grid),dtype=torch.bool,device=device)
        for x,y in exit_pos: self.exit_mask[y,x]=True
        # агенты
        self.N=sum(c for c,_ in agent_specs)
        self.size=torch.empty(self.N,device=device)
        self.speed=torch.empty(self.N,device=device)
        i=0
        for cnt,sz in agent_specs:
            for _ in range(cnt):
                self.size[i]=sz/5.; self.speed[i]=random.choice([3,4])/4.; i+=1
        self.knows_exit=torch.zeros(self.N,dtype=torch.bool,device=device); self.knows_exit[0]=True
        # runtime-тензоры
        self.positions=torch.zeros((self.N,2),dtype=torch.long,device=device)
        self.alive=torch.ones(self.N,dtype=torch.bool,device=device)
        self.health=torch.full((self.N,),100.,device=device)
        self.fire_mask=torch.zeros((grid,grid),dtype=torch.float32,device=device)
        self.reset()

    def reset(self):
        self.alive.fill_(True); self.health.fill_(100.)
        self.positions[:,0]=torch.randint(0,self.grid,(self.N,),device=device)
        self.positions[:,1]=torch.randint(0,self.grid,(self.N,),device=device)
        maxc=self.grid-(self.size*5).round().long()
        self.positions[:,0]=torch.min(self.positions[:,0],maxc)
        self.positions[:,1]=torch.min(self.positions[:,1],maxc)
        self.fire_mask.zero_()
        while int(self.fire_mask.sum())<self.num_fires:
            x=random.randint(0,self.grid-1); y=random.randint(0,self.grid-1)
            if (x,y) not in self.exit_positions: self.fire_mask[y,x]=1.

def stack_envs(envs):
    t=lambda attr: torch.stack([getattr(e,attr) for e in envs],0)
    first=envs[0]
    return (t("positions"), t("alive"), t("knows_exit"), t("health"),
            t("size"), t("speed"), t("fire_mask"), t("exit_mask"))
