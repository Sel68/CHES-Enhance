"""CHES agent in pure JAX: per-node encoder -> 1-D CNN across nodes (spatial) -> GRU over the
sequence of scheduled tasks (temporal) -> per-node actor / twin-critic heads. Masked discrete SAC
(paper Sec. IV, Alg. 1-2). One parameter set per (policy, disk, seed); everything is vmapped
over that leading axis, so all 12 agents train in one compiled program.
"""
import jax
import jax.numpy as jnp
import optax

HP = dict(width=64, gru=64, history=4, batch=256, gamma=0.9, tau=0.005, actor_lr=1e-4, critic_lr=3e-4,
          alpha_lr=3e-4, alpha0=0.05, ent_ratio=0.1, clip=5.0, decompose=True)


def init_net(key, F, width, gru):
    ks = iter(jax.random.split(key, 16))
    w = lambda i, o, g=1.: jax.random.normal(next(ks), (i, o)) * g * jnp.sqrt(2. / i)
    return dict(e1=w(F, width), e1b=jnp.zeros(width), e2=w(width, width), e2b=jnp.zeros(width),
                c1=jnp.stack([w(width, width, .6) for _ in range(3)]), c1b=jnp.zeros(width),
                c2=jnp.stack([w(width, width, .6) for _ in range(3)]), c2b=jnp.zeros(width),
                gx=w(2 * width, 3 * gru, .7), gh=w(gru, 3 * gru, .7), gb=jnp.zeros(3 * gru),
                h1=w(2 * width + gru, 64), h1b=jnp.zeros(64), h2=w(64, 16), h2b=jnp.zeros(16),
                out=w(16, 1, .05), outb=jnp.zeros(1), rej=w(gru, 1, .05), rejb=jnp.zeros(1))


def conv(x, k, b):
    """Kernel-3 1-D convolution across the node axis (nodes ordered CPU/GPU, then by x position)."""
    p = jnp.pad(x, [(0, 0)] * (x.ndim - 2) + [(1, 1), (0, 0)])
    return p[..., :-2, :] @ k[0] + p[..., 1:-1, :] @ k[1] + p[..., 2:, :] @ k[2] + b


def net(p, obs, valid):
    """obs [H,N,F], valid [H] -> logits/Q [N+1] (last = forced rejection)."""
    e = jax.nn.relu(obs @ p["e1"] + p["e1b"])
    e = jax.nn.relu(e @ p["e2"] + p["e2b"])
    pooled = jnp.concatenate([e.mean(1), e.max(1)], -1)                     # [H, 2W] per-step cluster summary
    c = jax.nn.relu(conv(e[-1], p["c1"], p["c1b"]))                         # spatial CNN on the current step
    c = jax.nn.relu(conv(c, p["c2"], p["c2b"])) + e[-1]

    def gru(h, inp):
        x, ok = inp
        xr, xz, xn = jnp.split(x @ p["gx"] + p["gb"], 3)
        hr, hz, hn = jnp.split(h @ p["gh"], 3)
        r, z = jax.nn.sigmoid(xr + hr), jax.nn.sigmoid(xz + hz)
        h2 = (1 - z) * jnp.tanh(xn + r * hn) + z * h
        return jnp.where(ok, h2, h), None
    h, _ = jax.lax.scan(gru, jnp.zeros(p["gh"].shape[0]), (pooled, valid))
    N = obs.shape[1]
    z = jnp.concatenate([c, e[-1], jnp.broadcast_to(h, (N, h.shape[0]))], -1)
    z = jax.nn.relu(z @ p["h1"] + p["h1b"])
    z = jax.nn.relu(z @ p["h2"] + p["h2b"])
    return jnp.concatenate([(z @ p["out"] + p["outb"])[:, 0], h @ p["rej"] + p["rejb"]])


bnet = jax.vmap(net, in_axes=(None, 0, 0))


def masked(logits, mask):
    logp = jax.nn.log_softmax(jnp.where(mask, logits, -1e9), -1)
    return jnp.exp(logp) * mask, jnp.where(mask, logp, 0.)


def optimizers(hp=HP):
    a = optax.chain(optax.clip_by_global_norm(hp["clip"]), optax.adam(hp["actor_lr"]))
    c = optax.chain(optax.clip_by_global_norm(hp["clip"]), optax.adam(hp["critic_lr"]))
    return a, c, optax.adam(hp["alpha_lr"])


def init_agent(key, F, hp=HP):
    ka, k1, k2 = jax.random.split(key, 3)
    actor = init_net(ka, F, hp["width"], hp["gru"])
    critics = (init_net(k1, F, hp["width"], hp["gru"]), init_net(k2, F, hp["width"], hp["gru"]))
    oa, oc, oal = optimizers(hp)
    log_alpha = jnp.log(jnp.array(hp["alpha0"], jnp.float32))
    return dict(actor=actor, critics=critics, targets=critics, log_alpha=log_alpha,
                oa=oa.init(actor), oc=oc.init(critics), oal=oal.init(log_alpha))


def sac_update(ag, b, hp=HP):
    """One masked discrete-SAC step (paper Eq. 23-28). b: s/s2 [B,H,N,F], v/v2 [B,H], m/m2 [B,A], a, d,
    R/R2 [B,A] = immediate reward of every action (known reward model, Eq. 18).

    decompose=True: Q(s,a) = R(s,a) + F(s,a); the critics learn only the future part F with target
    gamma * V(s'), V(s') = sum_a' pi(a'|s') [R(s',a') + min F_targ(s',a') - alpha log pi(a'|s')].
    This is an exact rewrite of the soft Bellman equation that removes the immediate reward from the
    regression (the action gaps between similar nodes are tiny compared with Q, so learning them through
    noisy bootstrapped targets makes the policy drift). decompose=False is the plain SAC critic.
    """
    oa, oc, oal = optimizers(hp)
    alpha = jnp.exp(ag["log_alpha"])
    k = 1. if hp["decompose"] else 0.
    p2, lp2 = masked(bnet(ag["actor"], b["s2"], b["v2"]), b["m2"])
    q2 = k * b["R2"] + jnp.minimum(bnet(ag["targets"][0], b["s2"], b["v2"]), bnet(ag["targets"][1], b["s2"], b["v2"]))
    v2 = jnp.sum(p2 * (jnp.where(b["m2"], q2, 0.) - alpha * lp2), -1)
    r = jnp.take_along_axis(b["R"], b["a"][:, None], -1)[:, 0]
    y = jax.lax.stop_gradient((1 - k) * r + hp["gamma"] * (1 - b["d"]) * v2)

    def critic_loss(cs):
        qs = [bnet(c, b["s"], b["v"]) for c in cs]
        qa = [jnp.take_along_axis(q, b["a"][:, None], -1)[:, 0] for q in qs]
        return sum(optax.huber_loss(q, y, delta=1.).mean() for q in qa), (qa[0].mean() + k * r.mean(), qs)
    (closs, (qmean, qs_all)), cg = jax.value_and_grad(critic_loss, has_aux=True)(ag["critics"])
    cu, oc_s = oc.update(cg, ag["oc"], ag["critics"])
    critics = optax.apply_updates(ag["critics"], cu)
    q = jax.lax.stop_gradient(k * b["R"] + jnp.minimum(*qs_all))         # pre-update critics (saves 2 passes)

    def actor_loss(ap):
        p, lp = masked(bnet(ap, b["s"], b["v"]), b["m"])
        ent = -jnp.sum(p * lp, -1)
        return jnp.mean(jnp.sum(p * (alpha * lp - jnp.where(b["m"], q, 0.)), -1)), ent
    (aloss, ent), ag_ = jax.value_and_grad(actor_loss, has_aux=True)(ag["actor"])
    au, oa_s = oa.update(ag_, ag["oa"], ag["actor"])
    actor = optax.apply_updates(ag["actor"], au)
    target_ent = hp["ent_ratio"] * jnp.log(jnp.maximum(b["m"].sum(-1), 1))
    gal = jnp.mean(ent - target_ent)                                     # d/dlog_alpha of log_alpha*(H - H_target)
    alu, oal_s = oal.update(gal, ag["oal"], ag["log_alpha"])
    log_alpha = jnp.clip(optax.apply_updates(ag["log_alpha"], alu), -12., 1.)
    targets = jax.tree.map(lambda t, c: (1 - hp["tau"]) * t + hp["tau"] * c, ag["targets"], critics)
    new = dict(actor=actor, critics=critics, targets=targets, log_alpha=log_alpha, oa=oa_s, oc=oc_s, oal=oal_s)
    stats = dict(critic_loss=closs, actor_loss=aloss, alpha=jnp.exp(log_alpha), entropy=ent.mean(), q=qmean)
    return new, stats


# ------------------------------------------------------------------ on-device replay buffer
def init_buffer(E, L, N, F, A):
    return dict(obs=jnp.zeros((E, L, N, F), jnp.float16), mask=jnp.zeros((E, L, A), bool),
                act=jnp.zeros((E, L), jnp.int32), rew=jnp.zeros((E, L, A)), done=jnp.zeros((E, L), bool),
                ep=jnp.full((E, L), -1, jnp.int32))


def buffer_write(buf, w, obs, mask, act, rew, done, ep):
    i = w % buf["act"].shape[1]
    upd = lambda x, y: x.at[:, i].set(y.astype(x.dtype))
    return dict(obs=upd(buf["obs"], obs), mask=upd(buf["mask"], mask), act=upd(buf["act"], act),
                rew=upd(buf["rew"], rew), done=upd(buf["done"], done), ep=upd(buf["ep"], ep))


def buffer_sample(buf, w, key, B, H):
    """Uniform transitions with their H-step histories; windows never cross episode boundaries."""
    E, L = buf["act"].shape
    ke, kt = jax.random.split(key)
    e = jax.random.randint(ke, (B,), 0, E)
    lo = jnp.maximum(w - L + H, 0)
    t = jax.random.randint(kt, (B,), lo, jnp.maximum(w - 1, lo + 1))      # t+1 must already be written
    pos = (t[:, None] + jnp.arange(-H + 1, 2)) % L                        # H+1 positions: t-H+1 .. t+1
    obs = buf["obs"][e[:, None], pos].astype(jnp.float32)
    ep = buf["ep"][e[:, None], pos]
    tt, t1 = t % L, (t + 1) % L
    return dict(s=obs[:, :-1], s2=obs[:, 1:], v=ep[:, :-1] == ep[:, -2:-1], v2=ep[:, 1:] == ep[:, -1:],
                m=buf["mask"][e, tt], m2=buf["mask"][e, t1], a=buf["act"][e, tt], R=buf["rew"][e, tt], R2=buf["rew"][e, t1],
                d=buf["done"][e, tt].astype(jnp.float32))
