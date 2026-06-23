"""
channels/physics/terrain_heightmap.py
M788 — Terrain heightmap with SPH fluid over procedural landscape.

Perlin fBm generates a continuous elevation field.  SPH boundary particles
line the surface; fluid particles flow downhill under gravity, pressure,
and viscosity, pooling in detected valleys.

Upstream: lygia snoise (Perlin basis), Müller 2003 (SPH kernels),
Stam 1999 (stable fluids), webgl-water (GPU surface rendering).
"""
from __future__ import annotations
import json, math, os
from dataclasses import dataclass
from typing import Any

# ── Perlin noise (shared basis with wind_field.py) ─────────────────────────
_PERM = list(range(256))
_s = 42
for _i in range(255, 0, -1):
    _s = (_s * 1103515245 + 12345) & 0x7FFFFFFF
    _j = _s % (_i + 1)
    _PERM[_i], _PERM[_j] = _PERM[_j], _PERM[_i]
_PERM *= 2
_G2 = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]

def _fade(t): return t*t*t*(t*(t*6-15)+10)
def _lerp(a,b,t): return a+t*(b-a)

def _dg(ix,iy,dx,dy):
    g=_G2[_PERM[_PERM[ix&255]+(iy&255)]%8]; return g[0]*dx+g[1]*dy

def perlin2(x,y):
    xi,yi=int(math.floor(x)),int(math.floor(y)); xf,yf=x-xi,y-yi
    u,v=_fade(xf),_fade(yf)
    return _lerp(_lerp(_dg(xi,yi,xf,yf),_dg(xi+1,yi,xf-1,yf),u),
                 _lerp(_dg(xi,yi+1,xf,yf-1),_dg(xi+1,yi+1,xf-1,yf-1),u),v)

def fbm2(x,y,octaves=5,lac=2.0,gain=0.5):
    v=a=0.0; amp=1.0; f=1.0
    for _ in range(octaves):
        v+=amp*perlin2(x*f,y*f); f*=lac; amp*=gain
    return v

# ── Configuration ──────────────────────────────────────────────────────────
@dataclass
class TerrainConfig:
    domain_x0: float = 180.0;  domain_x1: float = 420.0
    domain_y_base: float = 600.0;  domain_y_top: float = 100.0
    terrain_samples: int = 128
    amplitude: float = 140.0;  frequency: float = 0.012
    base_height: float = 0.35
    seed_ox: float = 7.31;  seed_oy: float = 3.14
    boundary_spacing: float = 3.0;  boundary_layers: int = 2
    n_fluid: int = 120;  particle_mass: float = 1.0
    rest_density: float = 10.0;  gas_constant: float = 50.0
    smoothing_radius: float = 16.0;  viscosity_mu: float = 80.0
    gravity: float = 60.0;  dt: float = 1/60
    damping: float = 0.92;  restitution: float = 0.2

# ── Terrain ────────────────────────────────────────────────────────────────
class Terrain:
    def __init__(self, cfg=None):
        self.cfg = cfg or TerrainConfig()
        c = self.cfg
        dx = (c.domain_x1-c.domain_x0)/max(c.terrain_samples-1,1)
        hr = c.domain_y_base - c.domain_y_top
        self.profile = []
        for i in range(c.terrain_samples):
            x = c.domain_x0 + i*dx
            n = fbm2(x*c.frequency + c.seed_ox, c.seed_oy)
            y = c.domain_y_base - hr*c.base_height - n*c.amplitude
            self.profile.append((x, y))

    def height_at(self, x):
        c = self.cfg
        if x <= self.profile[0][0]: return self.profile[0][1]
        if x >= self.profile[-1][0]: return self.profile[-1][1]
        t = (x-c.domain_x0)/(c.domain_x1-c.domain_x0)
        idx = t*(len(self.profile)-1); i = min(int(idx), len(self.profile)-2)
        return _lerp(self.profile[i][1], self.profile[i+1][1], idx-i)

    def normal_at(self, x):
        hl, hr = self.height_at(x-1), self.height_at(x+1)
        nx, ny = -(hr-hl), 2.0
        ln = math.hypot(nx, ny)+1e-12
        return nx/ln, ny/ln

    def find_valleys(self):
        valleys = []
        for i in range(1, len(self.profile)-1):
            _, yp = self.profile[i-1]; xc, yc = self.profile[i]; _, yn = self.profile[i+1]
            if yc > yp and yc > yn:
                rim = min(yp, yn)
                valleys.append({"x":round(xc,2),"y":round(yc,2),
                                "rim_y":round(rim,2),"capacity":round(abs(yc-rim),2)})
        return valleys

# ── SPH kernels (Müller et al. 2003) ──────────────────────────────────────
def _poly6(r_sq, h):
    h2 = h*h
    if r_sq >= h2: return 0.0
    d = h2-r_sq
    return (4/(math.pi*h2**4)) * d*d*d

def _spiky_grad(rx, ry, r, h):
    if r >= h or r < 1e-6: return 0.0, 0.0
    c = (-10/(math.pi*h**5))*(h-r)**2/r
    return c*rx, c*ry

def _visc_lap(r, h):
    return 0.0 if r >= h else (20/(3*math.pi*h**5))*(h-r)

# ── SPH simulation ────────────────────────────────────────────────────────
class SPHFluid:
    def __init__(self, terrain, cfg=None):
        self.terrain = terrain
        self.cfg = cfg or terrain.cfg
        self.tick_count = 0
        # Particle arrays: x, y, vx, vy, density, pressure, fx, fy, boundary
        self.px=[]; self.py=[]; self.pvx=[]; self.pvy=[]
        self.pd=[]; self.pp=[]; self.pfx=[]; self.pfy=[]; self.pb=[]
        self._seed_boundary()
        self._seed_fluid()

    def _add(self, x, y, bnd=False):
        self.px.append(x); self.py.append(y)
        self.pvx.append(0.0); self.pvy.append(0.0)
        self.pd.append(0.0); self.pp.append(0.0)
        self.pfx.append(0.0); self.pfy.append(0.0)
        self.pb.append(bnd)

    def _seed_boundary(self):
        c = self.cfg; prof = self.terrain.profile; accum = 0.0
        for i in range(len(prof)-1):
            x0,y0 = prof[i]; x1,y1 = prof[i+1]
            seg = math.hypot(x1-x0, y1-y0)
            if seg < 1e-6: continue
            while accum <= seg:
                t = accum/seg
                px, py = _lerp(x0,x1,t), _lerp(y0,y1,t)
                self._add(px, py, True)
                nx, ny = self.terrain.normal_at(px)
                for layer in range(1, c.boundary_layers+1):
                    off = layer*c.boundary_spacing
                    self._add(px-nx*off, py-ny*off, True)
                accum += c.boundary_spacing
            accum -= seg

    def _seed_fluid(self):
        c = self.cfg
        cols = int(math.sqrt(c.n_fluid*2)); rows = max(1, c.n_fluid//cols)
        dx = (c.domain_x1-c.domain_x0)/max(cols,1); count = 0
        for col in range(cols):
            for row in range(rows):
                if count >= c.n_fluid: break
                x = c.domain_x0 + (col+0.5)*dx
                ground = self.terrain.height_at(x)
                y = ground - c.smoothing_radius*(row+1)*1.5
                if y < c.domain_y_top: y = c.domain_y_top + c.smoothing_radius*0.5
                self._add(x, y); count += 1

    def step(self):
        c = self.cfg; h = c.smoothing_radius; h2 = h*h; n = len(self.px)
        # Density + pressure
        for i in range(n):
            self.pd[i] = 0.0
            for j in range(n):
                dx = self.px[i]-self.px[j]; dy = self.py[i]-self.py[j]
                self.pd[i] += c.particle_mass * _poly6(dx*dx+dy*dy, h)
            self.pd[i] = max(self.pd[i], 1e-6)
            self.pp[i] = c.gas_constant * (self.pd[i]-c.rest_density)
        # Forces
        for i in range(n):
            if self.pb[i]: continue
            self.pfx[i] = 0.0; self.pfy[i] = c.gravity*self.pd[i]
            for j in range(n):
                if i == j: continue
                dx = self.px[i]-self.px[j]; dy = self.py[i]-self.py[j]
                r = math.hypot(dx, dy)
                if r >= h or r < 1e-6: continue
                # Pressure
                gx, gy = _spiky_grad(dx, dy, r, h)
                pt = -c.particle_mass*(self.pp[i]+self.pp[j])*0.5/max(self.pd[j],1e-6)
                self.pfx[i] += pt*gx; self.pfy[i] += pt*gy
                # Viscosity (fluid-fluid only)
                if not self.pb[j]:
                    lap = _visc_lap(r, h)
                    vt = c.viscosity_mu*c.particle_mass/max(self.pd[j],1e-6)
                    self.pfx[i] += vt*(self.pvx[j]-self.pvx[i])*lap
                    self.pfy[i] += vt*(self.pvy[j]-self.pvy[i])*lap
        # Integrate
        dt = c.dt
        for i in range(n):
            if self.pb[i]: continue
            ir = 1.0/max(self.pd[i], 1e-6)
            self.pvx[i] = (self.pvx[i]+self.pfx[i]*ir*dt)*c.damping
            self.pvy[i] = (self.pvy[i]+self.pfy[i]*ir*dt)*c.damping
            self.px[i] += self.pvx[i]*dt; self.py[i] += self.pvy[i]*dt
            # Terrain collision
            gy = self.terrain.height_at(self.px[i])
            if self.py[i] > gy:
                nx, ny = self.terrain.normal_at(self.px[i])
                self.py[i] = gy - 1.0
                vn = self.pvx[i]*nx + self.pvy[i]*ny
                if vn < 0:
                    self.pvx[i] -= (1+c.restitution)*vn*nx
                    self.pvy[i] -= (1+c.restitution)*vn*ny
            # Domain walls
            if self.px[i] < c.domain_x0:
                self.px[i] = c.domain_x0+1; self.pvx[i] = abs(self.pvx[i])*c.restitution
            elif self.px[i] > c.domain_x1:
                self.px[i] = c.domain_x1-1; self.pvx[i] = -abs(self.pvx[i])*c.restitution
            if self.py[i] < c.domain_y_top:
                self.py[i] = c.domain_y_top+1; self.pvy[i] = abs(self.pvy[i])*c.restitution
        self.tick_count += 1

    def detect_pools(self):
        valleys = self.terrain.find_valleys(); h = self.cfg.smoothing_radius
        pools = []
        for v in valleys:
            vx = v["x"]
            near = [i for i in range(len(self.px))
                    if not self.pb[i] and abs(self.px[i]-vx)<h*4 and abs(self.pvy[i])<20]
            if len(near) < 3: continue
            wy = min(self.py[i] for i in near)
            pools.append({"valley_x":v["x"],"valley_y":v["y"],"rim_y":v["rim_y"],
                          "water_level_y":round(wy,2),"particle_count":len(near),
                          "fill_ratio":round(min(1.0,(v["y"]-wy)/max(v["capacity"],1)),3)})
        return pools

    def snapshot(self):
        c = self.cfg
        bnd = [{"x":round(self.px[i],2),"y":round(self.py[i],2)}
               for i in range(len(self.px)) if self.pb[i]]
        fld = [{"x":round(self.px[i],2),"y":round(self.py[i],2),
                "vx":round(self.pvx[i],3),"vy":round(self.pvy[i],3),
                "density":round(self.pd[i],1)}
               for i in range(len(self.px)) if not self.pb[i]]
        return {"tick":self.tick_count,
                "terrain_profile":[{"x":round(x,2),"y":round(y,2)} for x,y in self.terrain.profile],
                "valleys":self.terrain.find_valleys(),
                "boundary_particles":bnd,"fluid_particles":fld,
                "pools":self.detect_pools(),
                "domain":{"x0":c.domain_x0,"x1":c.domain_x1,
                          "y_top":c.domain_y_top,"y_base":c.domain_y_base}}

# ── Export / CLI ───────────────────────────────────────────────────────────
_PHYSICS_DIR = os.path.dirname(os.path.abspath(__file__))

def export(ticks=30):
    cfg = TerrainConfig(); t = Terrain(cfg); sim = SPHFluid(t, cfg)
    for _ in range(ticks): sim.step()
    return sim.snapshot()

if __name__ == "__main__":
    print("="*64)
    print("M788 Terrain Heightmap — SPH fluid over procedural landscape")
    print("="*64)
    snap = export(ticks=30)
    out = os.path.join(_PHYSICS_DIR, "terrain_heightmap.json")
    with open(out, "w") as f: json.dump(snap, f, indent=2)
    nb = len(snap["boundary_particles"]); nf = len(snap["fluid_particles"])
    print(f"\nWrote {out}")
    print(f"  terrain samples : {len(snap['terrain_profile'])}")
    print(f"  boundary particles: {nb}")
    print(f"  fluid particles   : {nf}")
    print(f"  valleys detected  : {len(snap['valleys'])}")
    print(f"  pools formed      : {len(snap['pools'])}")
    for p in snap["pools"]:
        print(f"    basin x={p['valley_x']:.0f}  fill={p['fill_ratio']:.1%}  n={p['particle_count']}")
    print(f"  ticks: {snap['tick']}")
    print("✓ complete")
