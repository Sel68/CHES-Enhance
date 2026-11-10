"""Event-driven heterogeneous edge simulator. No learning dependencies."""
from copy import deepcopy
import math
import numpy as np
from data import DISKS, POLICIES


NODE_FEATURES = 18


class Env:
    def __init__(self, catalogue, request_trace, policy="LRU", disk="tight", record=False, replay_slots=200):
        if policy not in POLICIES or disk not in DISKS:
            raise ValueError("Unknown eviction policy or disk regime")
        self.world, self.trace = catalogue, request_trace
        self.policy, self.disk = policy, disk
        self.disk_scale = catalogue.get("disk_scales", DISKS)[disk]
        self.record, self.replay_slots = record, replay_slots
        self.images = catalogue["images"]
        self.nodes = deepcopy(catalogue["nodes"])
        total = sum(i["size"] for i in self.images)
        for n in self.nodes:
            n.update(capacity=max(max(i["size"] for i in self.images), total * self.disk_scale),
                     cache={}, active=[], download_end=0., clock=0.)
        self.time, self.cursor, self.access = 0., 0, 0
        self.jobs, self.results, self.frames = {}, {}, []
        self.evictions, self.downloads, self.download_bytes = 0, 0, 0.
        self.latency_cost = 0.
        self.demand = np.zeros(len(self.images), dtype=np.float64)
        self.last_slot = -1

    @property
    def task(self):
        return self.trace["tasks"][self.cursor] if self.cursor < len(self.trace["tasks"]) else None

    @staticmethod
    def used(n):
        return sum(c["size"] for c in n["cache"].values())

    @staticmethod
    def memory_used(n):
        return sum(j["task"]["memory"] for j in n["active"])

    def work(self, task, node):
        # Normalized compute units: GPU services cost 4x their work on CPU.
        return task["work"] * (4 if task["type"] == "GPU" and node["type"] == "CPU" else 1)

    def rate(self, n, count):
        penalty = (1 - self.world["config"]["gpu_contention"]) ** (count - 1) if n["type"] == "GPU" else 1.
        return n["speed"] * penalty / max(1, count)

    def advance(self, until):
        """Exact processor sharing; reallocate rates at starts and completions."""
        while self.time < until - 1e-9:
            running = [[j for j in n["active"] if j["ready"] <= self.time + 1e-8] for n in self.nodes]
            event = until
            for n, jobs in zip(self.nodes, running):
                for j in n["active"]:
                    if j["ready"] > self.time + 1e-8:
                        event = min(event, j["ready"])
                if jobs:
                    event = min(event, self.time + min(j["remaining"] for j in jobs) / self.rate(n, len(jobs)))
            dt = max(0., event - self.time)
            self.latency_cost += dt * sum(len(n["active"]) for n in self.nodes)
            for n, jobs in zip(self.nodes, running):
                for j in jobs:
                    j["remaining"] -= dt * self.rate(n, len(jobs))
            self.time = event
            for n in self.nodes:
                for j in list(n["active"]):
                    if j["ready"] <= self.time + 1e-8 and j["remaining"] <= 1e-7:
                        n["active"].remove(j)
                        n["cache"][j["task"]["image"]]["pins"] -= 1
                        self._finish(j, self.time)

    def _finish(self, job, finish):
        t, event = job["task"], job["event"]
        latency = finish - t["arrival"]
        event.update(finish=finish, latency=latency, deadline_met=finish <= t["deadline"] + 1e-8)
        self.results[t["id"]] = event

    def remaining_cost(self):
        """Drain current jobs in a private projection, without future requests.

        This is a state-only shaping potential, not a scheduling oracle. It uses
        only already admitted work, known pull completion times and node rates.
        """
        cost = 0.
        for n in self.nodes:
            jobs = [[j["ready"], j["remaining"]] for j in n["active"]]
            now = self.time
            while jobs:
                running = [j for j in jobs if j[0] <= now + 1e-8]
                future = [j[0] for j in jobs if j[0] > now + 1e-8]
                event = min(future, default=float("inf"))
                if running:
                    rate = self.rate(n, len(running))
                    event = min(event, now + min(j[1] for j in running) / rate)
                elapsed = max(0., event - now)
                cost += len(jobs) * elapsed
                for j in running:
                    j[1] -= elapsed * rate
                now = event
                jobs = [j for j in jobs if not (j[0] <= now + 1e-8 and j[1] <= 1e-7)]
        return cost

    def feasible(self, task, n):
        if task["type"] == "CPU" and n["type"] != "CPU":
            return False
        if self.memory_used(n) + task["memory"] > n["memory"] + 1e-9:
            return False
        if task["image"] in n["cache"]:
            return True
        protected = sum(c["size"] for c in n["cache"].values() if c["pins"] or c["ready"] > self.time)
        return protected + self.images[task["image"]]["size"] <= n["capacity"] + 1e-9

    def mask(self):
        mask = np.array([self.feasible(self.task, n) for n in self.nodes] + [False])
        mask[-1] = not mask[:-1].any()  # Forced rejection only; never mask to all false.
        return mask

    def estimate(self, task, n):
        c = n["cache"].get(task["image"])
        download = max(0., c["ready"] - self.time) if c else max(0., n["download_end"] - self.time) + 8 * self.images[task["image"]]["size"] / n["bandwidth"]
        distance = math.hypot(task["x"] - n["x"], task["y"] - n["y"])
        upload = 8 * task["data"] / (n["bandwidth"] / (1 + (distance / 35) ** 2))
        compute = self.work(task, n) / self.rate(n, len(n["active"]) + 1)
        return max(download, upload) + compute, download, upload

    def observation(self):
        """Per-node state + full cache metadata + causal request-frequency features."""
        t = self.task
        if t is None:
            return np.zeros(self.obs_shape, np.float32)
        rows = []
        maxsize = max(i["size"] for i in self.images)
        for n in self.nodes:
            c = n["cache"].get(t["image"])
            latency, download, upload = self.estimate(t, n)
            features = [n["type"] == "GPU", n["speed"] / 16, n["bandwidth"] / 6,
                        1 - self.memory_used(n) / n["memory"], self.used(n) / n["capacity"],
                        len(n["active"]) / 20, n["capacity"] / sum(i["size"] for i in self.images),
                        bool(c and c["ready"] <= self.time), bool(c and c["ready"] > self.time),
                        min(download / 100, 10), min(latency / 100, 10), upload,
                        n["x"] / 100, n["y"] / 100,
                        self.images[t["image"]]["size"] / maxsize, t["type"] == "GPU",
                        (t["deadline"] - self.time) / 50, t["work"] / 5]
            if self.world.get("observation_version", 1) >= 2:
                features.append((self.trace["slots"] - 1 - t["slot"]) / max(1, self.trace["slots"] - 1))
            # For every image: resident, pin count, age, frequency, GDSF priority.
            for i in range(len(self.images)):
                entry = n["cache"].get(i)
                features.extend([bool(entry), min(entry["pins"] / 5, 1) if entry else 0,
                                 min((self.access - entry["last"]) / 100, 1) if entry else 0,
                                 min(entry["freq"] / 20, 1) if entry else 0,
                                 max(0., min((entry["key"] - n["clock"]) / 20, 1)) if entry else 0])
            features.extend(self.demand.tolist())
            features.extend([i == t["image"] for i in range(len(self.images))])
            rows.append(features)
        return np.asarray(rows, dtype=np.float32)

    @property
    def obs_shape(self):
        return (len(self.nodes), NODE_FEATURES + 7 * len(self.images) + int(self.world.get("observation_version", 1) >= 2))

    def heuristic(self, method, rng):
        valid = np.flatnonzero(self.mask())
        if valid[-1] == len(self.nodes):
            return int(valid[-1])
        if method == "Random":
            return int(rng.choice(valid))
        def score(i):
            n = self.nodes[i]
            if method == "Greedy_latency":
                return self.estimate(self.task, n)[0]
            if method == "Image_priority":
                c = n["cache"].get(self.task["image"])
                return (not (c and c["ready"] <= self.time), self.estimate(self.task, n)[0])
            if method == "Least_load":
                return len(n["active"])
            if method == "Greedy_resources":
                return -n["speed"] * (1 - self.memory_used(n) / n["memory"])
            raise ValueError(method)
        return int(min(valid, key=score))

    def victim(self, n):
        candidates = [(i, c) for i, c in n["cache"].items() if c["pins"] == 0 and c["ready"] <= self.time]
        def key(pair):
            i, c = pair
            return {"LRU": (c["last"], i), "FIFO": (c["inserted"], i),
                    "LFU": (c["freq"], c["last"], i), "GDSF": (c["key"], c["last"], i)}[self.policy]
        return min(candidates, key=key)[0]

    def step(self, action):
        t = self.task
        if t is None:
            raise RuntimeError("Trace exhausted")
        if not 0 <= action <= len(self.nodes) or not self.mask()[action]:
            raise ValueError(f"Infeasible action {action} for task {t['id']}")
        self.access += 1
        self.demand *= .97
        self.demand[t["image"]] += .03
        event = dict(task_id=t["id"], slot=t["slot"], arrival=t["arrival"], type=t["type"],
                     image=t["image"], origin=[t["x"], t["y"]], node=action if action < len(self.nodes) else -1,
                     deadline=t["deadline"], hit=False, coalesced=False, evicted=[],
                     download_start=None, download_finish=None, upload_finish=None,
                     finish=None, deadline_met=False, rejected=action == len(self.nodes))
        if action == len(self.nodes):
            # Finite failure latency: deadline allowance plus 50s penalty.
            self.latency_cost += t["deadline"] - t["arrival"] + 50
            self._finish(dict(task=t, event=event), t["deadline"] + 50)
        else:
            n = self.nodes[action]
            _, _, upload = self.estimate(t, n)
            image = t["image"]
            c = n["cache"].get(image)
            event["hit"] = bool(c and c["ready"] <= self.time)
            event["coalesced"] = bool(c and c["ready"] > self.time)
            if c is None:
                size = self.images[image]["size"]
                while self.used(n) + size > n["capacity"] + 1e-9:
                    victim = self.victim(n)
                    old = n["cache"].pop(victim)
                    if self.policy == "GDSF":
                        n["clock"] = max(n["clock"], old["key"])
                    event["evicted"].append(dict(node=action, image=victim, size=old["size"]))
                    self.evictions += 1
                start = max(self.time, n["download_end"])
                end = start + 8 * size / n["bandwidth"]
                c = dict(size=size, ready=end, start=start, inserted=self.access,
                         last=self.access, freq=0, pins=0, key=0.)
                n["cache"][image] = c
                n["download_end"] = end
                self.downloads += 1
                self.download_bytes += size
            c["freq"] += 1
            c["last"] = self.access
            # Retrieval cost includes fixed registry startup, avoids size cancellation.
            c["key"] = n["clock"] + c["freq"] * (1 + 8 * c["size"] / n["bandwidth"]) / c["size"]
            c["pins"] += 1
            event.update(download_start=c["start"], download_finish=c["ready"], upload_finish=self.time + upload)
            job = dict(task=t, event=event, remaining=self.work(t, n), ready=max(c["ready"], self.time + upload))
            n["active"].append(job)
        self.jobs[t["id"]] = event
        self.cursor += 1
        next_t = self.task
        if next_t is None or next_t["slot"] != t["slot"]:
            self.snapshot(t["slot"])
        if next_t:
            self.advance(next_t["arrival"])
        else:
            # Drain all admitted work, including downloads; no arbitrary truncation.
            while any(n["active"] for n in self.nodes):
                self.advance(self.time + 1000)
        return next_t is None

    def snapshot(self, slot):
        if self.record and slot < self.replay_slots:
            self.frames.append(dict(slot=slot, time=self.time, nodes=[
                [dict(image=i, size=c["size"], pins=c["pins"], start=c["start"], ready=c["ready"])
                 for i, c in n["cache"].items()] for n in self.nodes]))

    def metrics(self):
        events = list(self.results.values())
        if not events:
            raise ValueError("Evaluate at least one task")
        latency = np.array([e["latency"] for e in events])
        return dict(tasks=len(events), avg_latency=float(latency.mean()), p95_latency=float(np.quantile(latency, .95)),
                    reward=float(np.mean([(e["deadline"] - e["finish"]) / 50 for e in events])),
                    completion_ratio=float(np.mean([e["deadline_met"] for e in events])),
                    hit_rate=float(np.mean([e["hit"] for e in events])),
                    rejection_rate=float(np.mean([e["rejected"] for e in events])),
                    avg_download_wait=float(np.mean([max(0, (e["download_finish"] or e["arrival"]) - e["arrival"]) for e in events])),
                    evictions=self.evictions, downloads=self.downloads, downloaded_gb=self.download_bytes)

    def replay(self, method, seed):
        return dict(version=1, method=method, policy=self.policy, disk=self.disk, disk_scale=self.disk_scale,
                    seed=seed, trace=self.trace["id"], slot_seconds=self.world["config"]["slot_seconds"],
                    slots=min(self.replay_slots, self.trace["slots"]),
                    nodes=[{k: n[k] for k in ("id", "type", "x", "y", "capacity")} for n in self.nodes],
                    images=[{k: i[k] for k in ("id", "size", "type")} for i in self.images],
                    frames=self.frames, tasks=[e for e in self.jobs.values() if e["slot"] < self.replay_slots],
                    metrics=self.metrics())
