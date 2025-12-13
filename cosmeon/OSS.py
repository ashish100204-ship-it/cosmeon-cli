"""PROBLEM STATEMENT 3: ORBITAL SHARDED STORAGE (OSS)
Problem Summary
Sharding is essential for COSMEON’s orbital data layer. This problem requires building a storage service that
splits files into shards, distributes those shards across nodes, and reconstructs the file even if some shards are
missing.
Expectations
Participants must implement sharding, shard distribution, and reconstruction logic in a multi-node simulated
environment.
Minimum Requirements
• Split a file into N shards of equal or near-equal size.
• Store shards on multiple simulated satellite nodes.
• Maintain a shard map indicating which node stores which shard.
• Reconstruct the original file when requested.
• Demonstrate system behavior when certain shards or nodes are unavailable.
• Implement basic error handling and validation during reconstruction.
Optional Enhancements
• Simple parity shards or Reed-Solomon style erasure coding.
• Automatic shard healing when nodes rejoin.
• Real-time visualization of shard distribution and node status."""


from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import hashlib
import itertools
import random
import time
import json
import sys
from pathlib import Path
import typer

#  exception for reconstruction errors
class ReconstructionError(Exception):
    pass

# Exception for missing shards
class MissingShardsError(ReconstructionError):
    def __init__(self, missing_indices: List[int], details: str = ""):
        msg = f"Missing shards: {sorted(missing_indices)}"
        if details:
            msg += f" | Details: {details}"
        super().__init__(msg)
        self.missing = sorted(missing_indices)
        self.details = details

# Exception for corrupt shards
class CorruptShardError(ReconstructionError):
    def __init__(self, corrupt_list: List[Tuple[int, str]], details: str = ""):
        msg = f"Corrupt shards found at: {corrupt_list}"
        if details:
            msg += f" | Details: {details}"
        super().__init__(msg)
        self.corrupt = corrupt_list
        self.details = details

# Optional visualization libraries
try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Shard data structure with metadata
@dataclass(frozen=True)
class Shard:
    file_id: str
    shard_index: int
    total_shards: int
    data: bytes
    checksum: str
    kind: str
    group_id: Optional[int] = None

    # Create shard with checksum
    @staticmethod
    def make(file_id: str, shard_index: int, total_shards: int, data: bytes,
             kind: str = "data", group_id: Optional[int] = None) -> "Shard":
        checksum = hashlib.sha256(data).hexdigest()
        return Shard(file_id, shard_index, total_shards, data, checksum, kind, group_id)

    # Serialize shard to dictionary
    def to_dict(self) -> Dict:
        return {"file_id": self.file_id, "shard_index": self.shard_index, "total_shards": self.total_shards,
                "data": self.data.hex(), "checksum": self.checksum, "kind": self.kind, "group_id": self.group_id}

    # Deserialize shard from dictionary
    @staticmethod
    def from_dict(d: Dict) -> "Shard":
        data = bytes.fromhex(d["data"])
        return Shard(d["file_id"], int(d["shard_index"]), int(d["total_shards"]), data,
                      d["checksum"], d["kind"], d.get("group_id"))

# Satellite node simulation
class Node:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.online = True
        self.storage: Dict[Tuple[str, int], Shard] = {}

    def store_shard(self, shard: Shard):
        self.storage[(shard.file_id, shard.shard_index)] = shard

    def get_shard(self, file_id: str, shard_index: int) -> Optional[Shard]:
        return self.storage.get((file_id, shard_index)) if self.online else None

    def has_shard(self, file_id: str, shard_index: int) -> bool:
        return (file_id, shard_index) in self.storage

    def to_dict(self) -> Dict:
        return {"node_id": self.node_id, "online": self.online,
                "storage": {f"{fid}:{idx}": s.to_dict() for (fid, idx), s in self.storage.items()}}

    @staticmethod
    def from_dict(d: Dict) -> "Node":
        node = Node(d["node_id"])
        node.online = d.get("online", True)
        node.storage = {}
        for key, s_dict in d.get("storage", {}).items():
            try:
                fid, idx = key.split(":")
                node.storage[(fid, int(idx))] = Shard.from_dict(s_dict)
            except (ValueError, KeyError):
                continue
        return node

    def __repr__(self):
        return f"Node(id={self.node_id}, online={self.online}, shards={len(self.storage)})"

# Orchestrates sharding, distribution, and repair
class ShardManager:
    def __init__(self, replication_factor: int = 2, group_size: int = 3, use_global_parity: bool = True):
        self.nodes: Dict[str, Node] = {}
        self.shard_map: Dict[str, Dict[str, List[str]]] = {}
        self.file_meta: Dict[str, Dict] = {}
        self.replication_factor = replication_factor
        self.group_size = group_size
        self.use_global_parity = use_global_parity
        self.logs: List[str] = []

    def add_node(self, node: Node):
        if node.node_id in self.nodes: raise ValueError(f"Duplicate node_id: {node.node_id}")
        self.nodes[node.node_id] = node

    def set_node_status(self, node_id: str, online: bool):
        if node_id not in self.nodes: raise KeyError(f"Unknown node_id: {node_id}")
        self.nodes[node_id].online = online
        self.logs.append(self._stamp(f"[NODE] {node_id} -> {'ONLINE' if online else 'OFFLINE'}"))

    # Split file into equal/near-equal shards
    def _split_file(self, file_id: str, file_bytes: bytes, num_data_shards: int) -> List[Shard]:
        L = len(file_bytes)
        base, extra = divmod(L, num_data_shards)
        shards, start = [], 0
        for i in range(num_data_shards):
            size = base + (1 if i < extra else 0)
            chunk = file_bytes[start:start + size]
            start += size
            shards.append(Shard.make(file_id, i, num_data_shards, chunk, "data"))
        return shards

    # XOR operation for parity calculation
    @staticmethod
    def _xor_bytes(buffers: List[bytes]) -> bytes:
        if not buffers: return b""
        max_len = max(len(b) for b in buffers)
        acc = bytearray(max_len)
        for b in buffers:
            for i, byte in enumerate(b): acc[i] ^= byte
        return bytes(acc)

    # Create local parity shards for groups
    def _make_local_parities(self, file_id: str, data_shards: List[Shard], start_idx: int) -> List[Shard]:
        parities = []
        for gid, i in enumerate(range(0, len(data_shards), self.group_size)):
            group = data_shards[i:i + self.group_size]
            parity_data = self._xor_bytes([s.data for s in group])
            parities.append(Shard.make(file_id, start_idx + gid, len(data_shards), parity_data, "local_parity", gid))
        return parities

    # Create global parity shard
    def _make_global_parity(self, file_id: str, data_shards: List[Shard], shard_idx: int) -> Shard:
        gp_data = self._xor_bytes([s.data for s in data_shards])
        return Shard.make(file_id, shard_idx, len(data_shards), gp_data, "global_parity")

    # Distribute file across nodes with replication
    def distribute_file(self, file_id: str, file_bytes: bytes, num_data_shards: int):
        if not self.nodes:
            raise RuntimeError("No nodes available")
        if file_id in self.shard_map:
            raise ValueError(f"File ID '{file_id}' already exists")

        if self.replication_factor > len(self.nodes):
            raise ValueError(f"Replication factor ({self.replication_factor}) cannot exceed available nodes ({len(self.nodes)})")

        # Create data and parity shards
        data_shards = self._split_file(file_id, file_bytes, num_data_shards)
        local_parities = self._make_local_parities(file_id, data_shards, len(data_shards))
        next_idx = len(data_shards) + len(local_parities)
        global_parity = self._make_global_parity(file_id, data_shards, next_idx) if self.use_global_parity else None

        all_shards = data_shards + local_parities + ([global_parity] if global_parity else [])

        shard_sizes = [len(s.data) for s in data_shards]

        # Store shard map and metadata
        self.shard_map[file_id] = {}
        self.file_meta[file_id] = {
            "num_data": len(data_shards),
            "num_local_parity": len(local_parities),
            "has_global_parity": bool(global_parity),
            "group_size": self.group_size,
            "shard_sizes": shard_sizes,
            "original_size": len(file_bytes),
        }

        # Round-robin distribution with replication
        node_ids = list(self.nodes.keys())
        rrs = [itertools.cycle(random.sample(node_ids, len(node_ids))) for _ in range(self.replication_factor)]

        # Place each shard on multiple nodes
        for shard in all_shards:
            targets: List[str] = []
            chosen = set()
            for rr in rrs:
                attempts = 0
                node_id = next(rr)
                while node_id in chosen and attempts < len(node_ids):
                    node_id = next(rr)
                    attempts += 1
                chosen.add(node_id)
                targets.append(node_id)
                self.nodes[node_id].store_shard(shard)

            self.shard_map[file_id][str(shard.shard_index)] = targets

        self.logs.append(self._stamp(
            f"[DISTRIBUTE] File '{file_id}' created with {len(data_shards)} data, {len(local_parities)} local, {1 if global_parity else 0} global shards."
        ))

    # Reconstruct file from shards with repair
    def reconstruct_file(self, file_id: str) -> bytes:
        if file_id not in self.shard_map:
            raise KeyError(f"Unknown file_id: {file_id}")

        meta = self.file_meta.get(file_id)
        if not meta:
            raise ReconstructionError(f"Missing metadata for file '{file_id}'")

        num_data = int(meta.get("num_data", 0))
        group_size = int(meta.get("group_size", 1))
        shard_sizes = meta.get("shard_sizes")
        original_size = meta.get("original_size")

        collected: Dict[int, Shard] = {}
        missing: List[int] = []
        available_local: Dict[int, Shard] = {}
        gp: Optional[Shard] = None

        # Collect available shards
        for idx_str, holders in self.shard_map[file_id].items():
            try:
                idx = int(idx_str)
            except ValueError:
                self.logs.append(self._stamp(f"[WARN] Malformed shard index key: {idx_str}"))
                continue

            shard = self._fetch_valid_shard(file_id, idx, holders)
            if shard:
                if shard.kind == "data":
                    collected[idx] = shard
                elif shard.kind == "local_parity":
                    gid = shard.group_id if shard.group_id is not None else (idx - num_data)
                    available_local[gid] = shard
                elif shard.kind == "global_parity":
                    gp = shard
            else:
                if idx < num_data:
                    missing.append(idx)

        if len(collected) == 0 and not available_local and not gp:
            raise MissingShardsError(missing, details="No available shards or parity to attempt reconstruction.")

        # Repair missing shards using local parity
        groups = {gid: [i for i in range(num_data) if i // group_size == gid] for gid in range(meta.get("num_local_parity", 0))}
        for gid, members in groups.items():
            missing_in_group = [i for i in members if i in missing]
            if len(missing_in_group) == 1 and gid in available_local:
                miss_idx = missing_in_group[0]
                present_data = [collected[i].data for i in members if i in collected]
                repaired_data = self._xor_bytes([available_local[gid].data] + present_data)
                if shard_sizes and len(shard_sizes) > miss_idx:
                    target_len = shard_sizes[miss_idx]
                    repaired_trimmed = repaired_data[:target_len]
                elif original_size is not None:
                    base, extra = divmod(original_size, num_data)
                    size = base + (1 if miss_idx < extra else 0)
                    repaired_trimmed = repaired_data[:size]
                else:
                    repaired_trimmed = repaired_data
                collected[miss_idx] = Shard.make(file_id, miss_idx, num_data, repaired_trimmed, "data")
                missing.remove(miss_idx)
                self.logs.append(self._stamp(f"[REPAIR-LOCAL] Shard {miss_idx} of '{file_id}' using local parity gid={gid}"))

        # Repair last missing shard using global parity
        if len(missing) == 1 and gp:
            miss_idx = missing[0]
            present_data = [collected[i].data for i in range(num_data) if i in collected]
            repaired_data = self._xor_bytes([gp.data] + present_data)
            if shard_sizes and len(shard_sizes) > miss_idx:
                target_len = shard_sizes[miss_idx]
                repaired_trimmed = repaired_data[:target_len]
            elif original_size is not None:
                base, extra = divmod(original_size, num_data)
                size = base + (1 if miss_idx < extra else 0)
                repaired_trimmed = repaired_data[:size]
            else:
                repaired_trimmed = repaired_data
            collected[miss_idx] = Shard.make(file_id, miss_idx, num_data, repaired_trimmed, "data")
            missing.clear()
            self.logs.append(self._stamp(f"[REPAIR-GLOBAL] Shard {miss_idx} of '{file_id}' using global parity"))

        if missing:
            detail_lines = []
            for m in sorted(missing):
                holders = self.shard_map[file_id].get(str(m), [])
                detail_lines.append(f"{m}: holders={holders}")
            detail = "; ".join(detail_lines)
            raise MissingShardsError(missing, details=detail)

        for i in range(num_data):
            if i not in collected:
                raise ReconstructionError(f"Inconsistent state: shard {i} missing after repair attempts")

        # Assemble final file
        full = b"".join(collected[i].data for i in range(num_data))
        if original_size is not None:
            return full[:original_size]
        return full

    # Fetch valid shard with checksum verification
    def _fetch_valid_shard(self, file_id: str, index: int, holders: List[str]) -> Optional[Shard]:
        corrupt_locations = []
        absent_locations = []
        for nid in holders:
            node = self.nodes.get(nid)
            if not node:
                absent_locations.append((nid, "unknown_node"))
                continue
            if not node.online:
                absent_locations.append((nid, "offline"))
                continue
            shard = node.get_shard(file_id, index)
            if shard is None:
                absent_locations.append((nid, "no_shard"))
                continue
            actual = hashlib.sha256(shard.data).hexdigest()
            if actual == shard.checksum:
                return shard
            else:
                corrupt_locations.append((index, nid))
                self.logs.append(self._stamp(f"[CORRUPT] Shard {index} on {nid} (expected {shard.checksum[:8]} got {actual[:8]})"))
        if corrupt_locations:
            self.logs.append(self._stamp(f"[FETCH] No valid shard {index}; corrupt copies at {corrupt_locations}; absent/ offline at {absent_locations}"))
        else:
            self.logs.append(self._stamp(f"[FETCH] No shard {index} available from holders; absent/ offline at {absent_locations}"))
        return None

    # Restore missing shards on a node
    def heal_node(self, node_id: str):
        if not self.nodes.get(node_id, None) or not self.nodes[node_id].online: return
        node, repaired = self.nodes[node_id], 0
        for fid, entries in self.shard_map.items():
            for idx_str, holders in entries.items():
                idx = int(idx_str)
                if node_id in holders and not node.has_shard(fid, idx):
                    source = self._fetch_valid_shard(fid, idx, [h for h in holders if h != node_id])
                    if source:
                        node.store_shard(source)
                        repaired += 1
                        self.logs.append(self._stamp(f"[HEAL-COPY] Node {node_id} restored shard {idx}"))
                    elif idx < self.file_meta[fid]["num_data"]:
                        try:
                            full_file = self.reconstruct_file(fid)
                            regen = self._split_file(fid, full_file, self.file_meta[fid]["num_data"])[idx]
                            node.store_shard(regen)
                            repaired += 1
                            self.logs.append(self._stamp(f"[HEAL-REPAIR] Node {node_id} restored shard {idx}"))
                        except Exception as e:
                            self.logs.append(self._stamp(f"[HEAL-FAIL] on {node_id} for shard {idx}: {e}"))
        self.logs.append(self._stamp(f"[HEAL] Node {node_id} finished. Restored {repaired} shards."))

    # Display shard distribution matrix
    def print_matrix(self, file_id: str):
        if file_id not in self.shard_map:
            typer.echo(f"File ID '{file_id}' not found in system.")
            return
        node_ids, shard_indices = sorted(self.nodes.keys()), sorted([int(k) for k in self.shard_map[file_id].keys()])
        meta, kinds = self.file_meta[file_id], {}
        for si in shard_indices:
            if si < meta["num_data"]: kinds[si] = "Data"
            elif si < meta["num_data"] + meta["num_local_parity"]: kinds[si] = "Local"
            else: kinds[si] = "Global"

        header = f"{'Idx':>3} | {'Kind':<6} | " + " ".join(f"{nid:>6}" for nid in node_ids)
        typer.echo("\nShard Presence Matrix (X=Present, .=Absent, Red=Offline):")
        typer.echo(header + "\n" + "-" * len(header))
        for si in shard_indices:
            row = f"{si:>3} | {kinds[si]:<6} | "
            for nid in node_ids:
                mark = "X" if self.nodes[nid].has_shard(file_id, si) else "."
                color = typer.colors.RED if not self.nodes[nid].online else None
                row += typer.style(f"{mark:>6}", fg=color)
            typer.echo(row)

    def print_logs(self):
        if not self.logs: return
        typer.echo("\n" + "="*12 + " Logs " + "="*12)
        for log in self.logs: typer.echo(log)
        typer.echo("="*30)
        self.logs.clear()

    # Add timestamp to log message
    @staticmethod
    def _stamp(msg: str) -> str:
        return f"[{time.strftime('%H:%M:%S')}] {msg}"

    # Save system state to JSON
    def save_state(self, filepath: Path):
        data = {"nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
                "shard_map": self.shard_map, "file_meta": self.file_meta,
                "replication_factor": self.replication_factor, "group_size": self.group_size,
                "use_global_parity": self.use_global_parity}
        filepath.write_text(json.dumps(data, indent=2))

    # Load system state from JSON
    @classmethod
    def load_state(cls, filepath: Path) -> "ShardManager":
        data = json.loads(filepath.read_text())
        manager = cls(data.get("replication_factor", 2), data.get("group_size", 3), data.get("use_global_parity", True))
        manager.nodes = {nid: Node.from_dict(n_dict) for nid, n_dict in data.get("nodes", {}).items()}
        manager.shard_map = data.get("shard_map", {})
        manager.file_meta = data.get("file_meta", {})
        return manager

# CLI application
app = typer.Typer(help="A CLI for simulating a fault-tolerant Orbital Sharded Storage system.")

def load_manager(state_file: Path) -> ShardManager:
    """Loads the ShardManager from a state file."""
    if not state_file.exists():
        typer.echo(f"Error: State file not found at '{state_file}'")
        raise typer.Exit(code=1)
    return ShardManager.load_state(state_file)

def save_manager(manager: ShardManager, state_file: Path):
    """Saves the ShardManager to a state file."""
    manager.save_state(state_file)
    manager.print_logs()
    typer.echo(f"✅ State saved to '{state_file}'")

# Initialize storage system with nodes
@app.command()
def init(
    state_file: Path = typer.Option(..., "--state-file", "-f", help="Path to save the initial state file."),
    num_nodes: int = typer.Option(6, "--nodes", "-n", help="Number of satellite nodes to create."),
    repl_factor: int = typer.Option(2, "--repl", help="Replication factor for each shard."),
    group_size: int = typer.Option(3, "--group-size", help="Number of data shards per local parity group."),
    global_parity: bool = typer.Option(True, help="Whether to use a global parity shard.")
):
    
    random.seed(42)
    manager = ShardManager(repl_factor, group_size, global_parity)
    for i in range(num_nodes):
        manager.add_node(Node(f"SAT-{i}"))
    typer.echo(f"Initialized system with {num_nodes} nodes.")
    save_manager(manager, state_file)

# Display system status
@app.command()
def status(
    state_file: Path = typer.Option(..., "--state-file", "-f", help="Path to the state file to inspect."),
    file_id: Optional[str] = typer.Argument(None, help="Optional: File ID to show the shard matrix for.")
):
    """Displays the status of all nodes and an optional shard matrix for a file."""
    manager = load_manager(state_file)
    typer.echo("Node Status:")
    for node in manager.nodes.values():
        typer.echo(f"  - {node}")
    if file_id:
        manager.print_matrix(file_id)

# Distribute file into sharded storage
@app.command()
def distribute(
    state_file: Path = typer.Option(..., "--state-file", "-f", help="Path to the system state file (will be updated)."),
    local_file: Path = typer.Argument(..., help="Path to the local file to distribute."),
    file_id: str = typer.Option(..., "--file-id", help="A unique ID for the file within the system."),
    num_shards: int = typer.Option(6, "--shards", "-s", help="Number of data shards to split the file into.")
):
    """Distributes a local file into the sharded storage system."""
    manager = load_manager(state_file)
    if not local_file.exists():
        typer.echo(f"Error: Local file not found at '{local_file}'")
        raise typer.Exit(code=1)
    
    file_bytes = local_file.read_bytes()
    try:
        manager.distribute_file(file_id, file_bytes, num_shards)
        manager.print_matrix(file_id)
        save_manager(manager, state_file)
    except (ValueError, RuntimeError) as e:
        typer.echo(f"Error distributing file: {e}")
        raise typer.Exit(code=1)

# Reconstruct file from shards
@app.command()
def reconstruct(
    state_file: Path = typer.Option(..., "--state-file", "-f", help="Path to the system state file."),
    file_id: str = typer.Argument(..., help="The ID of the file to reconstruct."),
    output_file: Path = typer.Argument(..., help="Path to save the reconstructed file.")
):
   
    manager = load_manager(state_file)
    try:
        reconstructed_bytes = manager.reconstruct_file(file_id)
        output_file.write_bytes(reconstructed_bytes)
        manager.print_logs()
        typer.echo(f"✅ File '{file_id}' reconstructed successfully and saved to '{output_file}'")
        
        original_hash = hashlib.sha256(reconstructed_bytes).hexdigest()
        typer.echo(f"Reconstructed file SHA-256: {original_hash}")

    except (KeyError, RuntimeError, MissingShardsError, CorruptShardError, ReconstructionError) as e:
        manager.print_logs()
        typer.echo(f"Error reconstructing file: {e}")
        raise typer.Exit(code=1)

# Set node online/offline status
@app.command("set-status")
def set_node_status(
    state_file: Path = typer.Option(..., "--state-file", "-f", help="Path to the system state file (will be updated)."),
    node_id: str = typer.Argument(..., help="The ID of the node to update (e.g., 'SAT-1')."),
    online: bool = typer.Argument(..., help="The new status: 'True' for online, 'False' for offline.")
):
   
    manager = load_manager(state_file)
    try:
        manager.set_node_status(node_id, online)
        save_manager(manager, state_file)
    except KeyError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(code=1)

# Heal node by restoring missing shards
@app.command()
def heal(
    state_file: Path = typer.Option(..., "--state-file", "-f", help="Path to the system state file (will be updated)."),
    node_id: str = typer.Argument(..., help="The ID of the online node to heal.")
):
   
    manager = load_manager(state_file)
    if node_id not in manager.nodes or not manager.nodes[node_id].online:
        typer.echo(f"Error: Cannot heal node '{node_id}'. It is either unknown or offline.")
        raise typer.Exit(code=1)
    
    manager.heal_node(node_id)
    save_manager(manager, state_file)
if __name__ == "__main__":
    app()
