"""Pure JAX CNN + finite-history GRU discrete SAC, with masked twin critics."""
from collections import deque
import pickle
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax


def initialize(key, features, width, local_branch=False):
    keys = iter(jax.random.split(key, 12))
    def weight(shape, gain=1.):
        return jax.random.normal(next(keys), shape) * gain / np.sqrt(shape[-2])
    params = dict(project=weight((features, width)), project_b=jnp.zeros(width),
                conv=weight((3, width, width)), conv_b=jnp.zeros(width),
                gx=weight((2 * width, 3 * width)), gh=weight((width, 3 * width)), gb=jnp.zeros(3 * width),
                head=weight((2 * width, width)), head_b=jnp.zeros(width),
                out=weight((width, 1), .01), out_b=jnp.zeros(1),
                reject=weight((width, 1), .01), reject_b=jnp.zeros(1))
    if local_branch:
        params["local"] = jnp.zeros((min(18, features), 1))
    return params


def network(params, history):
    """[batch, history, node, feature] -> [batch, node + forced rejection]."""
    b, t, n, _ = history.shape
    width = params["project"].shape[1]
    x = jax.nn.silu(history @ params["project"] + params["project_b"])
    x = jax.lax.conv_general_dilated(x.reshape(b * t, n, width), params["conv"], (1,), "SAME",
                                    dimension_numbers=("NWC", "WIO", "NWC"))
    x = jax.nn.silu(x + params["conv_b"]).reshape(b, t, n, width)
    pooled = jnp.concatenate([x.mean(2), x.max(2)], -1)
    valid = jnp.any(history != 0, axis=(2, 3))
    def gru(h, inputs):
        inp, keep = inputs
        xp = inp @ params["gx"] + params["gb"]
        hp = h @ params["gh"]
        xr, xz, xh = jnp.split(xp, 3, -1)
        hr, hz, hh = jnp.split(hp, 3, -1)
        r, z = jax.nn.sigmoid(xr + hr), jax.nn.sigmoid(xz + hz)
        proposal = (1 - z) * jnp.tanh(xh + r * hh) + z * h
        h = jnp.where(keep[:, None], proposal, h)
        return h, None
    h, _ = jax.lax.scan(gru, jnp.zeros((b, width)), (pooled.swapaxes(0, 1), valid.T))
    fused = jnp.concatenate([x[:, -1], jnp.broadcast_to(h[:, None], (b, n, width))], -1)
    scores = (jax.nn.silu(fused @ params["head"] + params["head_b"]) @ params["out"] + params["out_b"])[..., 0]
    if "local" in params:
        # Preserve each node's immediate features alongside contextual CNN/GRU scores.
        scores += (history[:, -1, :, :params["local"].shape[0]] @ params["local"])[..., 0]
    # Forced rejection still needs a learned Q value to propagate future failures.
    reject = h @ params["reject"] + params["reject_b"]
    return jnp.concatenate([scores, reject], -1)


def probabilities(logits, mask):
    logp = jax.nn.log_softmax(jnp.where(mask, logits, -1e9), -1)
    return jnp.exp(logp), logp


class Replay:
    """Episode arrays avoid duplicated next states and stale stored GRU hidden states."""
    def __init__(self, capacity=30000, history=8):
        self.capacity, self.history = capacity, history
        self.episodes = deque()
        self.size = 0

    def add(self, states, masks, actions, rewards):
        length = len(actions)
        self.episodes.append((np.asarray(states, np.float16), np.asarray(masks, bool),
                              np.asarray(actions, np.int32), np.asarray(rewards, np.float32)))
        self.size += length
        while self.size > self.capacity and len(self.episodes) > 1:
            self.size -= len(self.episodes.popleft()[2])

    def sample(self, batch, rng):
        lengths = np.array([len(e[2]) for e in self.episodes])
        picks = rng.integers(self.size, size=batch)
        cumulative = np.cumsum(lengths)
        episode_ids = np.searchsorted(cumulative, picks, side="right")
        offsets = picks - np.r_[0, cumulative[:-1]][episode_ids]
        first = self.episodes[0][0]
        shape = (batch, self.history, *first.shape[1:])
        states, next_states = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
        masks, next_masks, actions, rewards, dones = [], [], [], [], []
        for i, (ei, at) in enumerate(zip(episode_ids, offsets)):
            s, m, a, r = self.episodes[ei]
            start = max(0, at - self.history + 1)
            chunk = s[start:at + 1]
            states[i, -len(chunk):] = chunk
            chunk = s[max(0, at - self.history + 2):at + 2]
            next_states[i, -len(chunk):] = chunk
            masks.append(m[at]); next_masks.append(m[at + 1])
            actions.append(a[at]); rewards.append(r[at]); dones.append(at == len(a) - 1)
        return dict(s=states, ns=next_states, m=np.array(masks), nm=np.array(next_masks),
                    a=np.array(actions), r=np.array(rewards), d=np.array(dones, np.float32))


class SAC:
    def __init__(self, obs_shape, seed=0, width=64, history=8, actor_lr=1e-4, critic_lr=3e-4, local_branch=False):
        self.obs_shape, self.width, self.history = tuple(obs_shape), width, history
        self.seed = seed
        keys = jax.random.split(jax.random.PRNGKey(seed), 3)
        self.actor, q1, q2 = [initialize(k, obs_shape[-1], width, local_branch) for k in keys]
        self.critics = (q1, q2)
        self.targets = self.critics
        self.log_alpha = jnp.array(np.log(.01), jnp.float32)
        self.actor_opt = optax.chain(optax.clip_by_global_norm(5), optax.adam(actor_lr))
        self.critic_opt = optax.chain(optax.clip_by_global_norm(5), optax.adam(critic_lr))
        self.alpha_opt = optax.adam(1e-4)
        self.actor_state = self.actor_opt.init(self.actor)
        self.critic_state = self.critic_opt.init(self.critics)
        self.alpha_state = self.alpha_opt.init(self.log_alpha)
        self.infer = jax.jit(lambda p, s, m: probabilities(network(p, s), m)[0])
        self.compiled_update = jax.jit(self._update)
        self.compiled_bc = jax.jit(self._bc)
        self.updates = 0

    def act(self, history, mask, rng=None):
        p = np.asarray(self.infer(self.actor, np.asarray(history, np.float32)[None], mask[None]))[0]
        p = p.astype(np.float64) * mask
        p /= p.sum()
        return int(np.argmax(p) if rng is None else rng.choice(len(p), p=p))

    def _bc(self, actor, opt_state, batch):
        def loss(p):
            _, logp = probabilities(network(p, batch["s"]), batch["m"])
            return -jnp.mean(jnp.take_along_axis(logp, batch["a"][:, None], -1))
        value, grads = jax.value_and_grad(loss)(actor)
        update, opt_state = self.actor_opt.update(grads, opt_state, actor)
        return optax.apply_updates(actor, update), opt_state, value

    def _update(self, actor, critics, targets, log_alpha, ast, cst, tst, batch):
        alpha = jnp.exp(log_alpha)
        p, lp = probabilities(network(actor, batch["ns"]), batch["nm"])
        target_q = jnp.minimum(network(targets[0], batch["ns"]), network(targets[1], batch["ns"]))
        target = jax.lax.stop_gradient(batch["r"] + .98 * (1 - batch["d"]) * jnp.sum(p * (target_q - alpha * lp), -1))
        def critic_loss(qs):
            selected = [jnp.take_along_axis(network(q, batch["s"]), batch["a"][:, None], -1)[:, 0] for q in qs]
            return sum(jnp.mean(optax.huber_loss(q, target, delta=1.)) for q in selected)
        closs, cg = jax.value_and_grad(critic_loss)(critics)
        cu, cst = self.critic_opt.update(cg, cst, critics)
        critics = optax.apply_updates(critics, cu)
        q = jax.lax.stop_gradient(jnp.minimum(network(critics[0], batch["s"]), network(critics[1], batch["s"])))
        def actor_loss(ap):
            p, lp = probabilities(network(ap, batch["s"]), batch["m"])
            entropy = -jnp.sum(p * lp, -1)
            return jnp.mean(jnp.sum(p * (alpha * lp - q), -1)), entropy.mean()
        (aloss, entropy), ag = jax.value_and_grad(actor_loss, has_aux=True)(actor)
        au, ast = self.actor_opt.update(ag, ast, actor)
        actor = optax.apply_updates(actor, au)
        target_entropy = .65 * jnp.mean(jnp.log(jnp.maximum(1, batch["m"].sum(-1))))
        tg = alpha * jax.lax.stop_gradient(entropy - target_entropy)
        tu, tst = self.alpha_opt.update(tg, tst, log_alpha)
        log_alpha = jnp.clip(optax.apply_updates(log_alpha, tu), -9.2, 0.)
        targets = jax.tree.map(lambda t, q: .995 * t + .005 * q, targets, critics)
        diagnostics = dict(critic_loss=closs, actor_loss=aloss, alpha=jnp.exp(log_alpha), entropy=entropy,
                           target_q=jnp.mean(target), grad_norm=optax.tree.norm(cg))
        return actor, critics, targets, log_alpha, ast, cst, tst, diagnostics

    def update(self, batch, imitation=False):
        if imitation:
            self.actor, self.actor_state, loss = self.compiled_bc(self.actor, self.actor_state, batch)
            return dict(bc_loss=float(loss))
        result = self.compiled_update(self.actor, self.critics, self.targets, self.log_alpha,
                                      self.actor_state, self.critic_state, self.alpha_state, batch)
        (self.actor, self.critics, self.targets, self.log_alpha,
         self.actor_state, self.critic_state, self.alpha_state, diagnostics) = result
        self.updates += 1
        diagnostics = {k: float(v) for k, v in diagnostics.items()}
        if not all(np.isfinite(v) for v in diagnostics.values()):
            raise FloatingPointError(f"Nonfinite SAC diagnostics: {diagnostics}")
        return diagnostics

    def save(self, path, metadata=None):
        payload = dict(obs_shape=self.obs_shape, seed=self.seed, width=self.width, history=self.history,
                       local_branch="local" in self.actor,
                       actor=self.actor, critics=self.critics, targets=self.targets, log_alpha=self.log_alpha,
                       actor_state=self.actor_state, critic_state=self.critic_state, alpha_state=self.alpha_state,
                       updates=self.updates, metadata=metadata or {})
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(pickle.dumps(jax.device_get(payload)))
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        # Checkpoints are local pickle files: load only trusted experiment outputs.
        p = pickle.loads(Path(path).read_bytes())
        obj = cls(p["obs_shape"], p["seed"], p["width"], p["history"], local_branch=p.get("local_branch", False))
        for name in ("actor", "critics", "targets", "log_alpha", "actor_state", "critic_state", "alpha_state", "updates"):
            setattr(obj, name, jax.tree.map(jnp.asarray, p[name]))
        obj.updates = int(obj.updates)
        obj.metadata = p["metadata"]
        return obj
