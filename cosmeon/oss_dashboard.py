# for visualizing and managing OSS.py ShardManager  run in terminal----  python oss_dashboard.py --state-file state.json --host 127.0.0.1 --port 8080
# Robust dashboard starter for OSS.py ShardManager
from flask import Flask, jsonify, render_template_string, request
from pathlib import Path
import sys
import traceback
import json
import argparse
import time

# Try to import ShardManager from OSS.py and show helpful error if it fails.
try:
    # OSS.py is expected to be in the same folder and define ShardManager
    from OSS import ShardManager
except Exception as e:
    print("ERROR: Failed importing ShardManager from OSS.py.", file=sys.stderr)
    traceback.print_exc()
    print("\nMake sure OSS.py is in the same directory and contains class ShardManager.", file=sys.stderr)
    sys.exit(1)

app = Flask(__name__)

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>OSS Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; background: #121212; color: #e8e8e8; padding: 16px; }
    table { border-collapse: collapse; width: 100%; margin-top: 12px; }
    th, td { padding: 8px 10px; border: 1px solid #333; text-align:left; }
    .online { background: #0a4; color: #001; }
    .offline { background: #a22; color: #200; }
    .small { font-size: 0.9em; color:#aaa; }
  </style>
</head>
<body>
  <h1>Orbital Sharded Storage — Dashboard</h1>
  <p class="small">Auto-refresh every 2s. Server time: <span id="time"></span></p>
  <div id="content">loading...</div>

<script>
async function load() {
  try {
    const r = await fetch('/state');
    const data = await r.json();
    let s = `<h2>Nodes (${Object.keys(data.nodes).length})</h2>`;
    s += '<table><tr><th>Node ID</th><th>Status</th><th>#Shards</th><th>Sample shards</th></tr>';
    for (const [nid, node] of Object.entries(data.nodes)) {
      const cls = node.online ? 'online' : 'offline';
      const shardCount = Object.keys(node.storage || {}).length;
      const sample = Object.keys(node.storage || {}).slice(0,4).join(', ');
      s += `<tr><td>${nid}</td><td class="${cls}">${node.online}</td><td>${shardCount}</td><td>${sample}</td></tr>`;
    }
    s += '</table>';
    s += `<h3>Files: ${Object.keys(data.shard_map || {}).length}</h3>`;
    document.getElementById('content').innerHTML = s;
    document.getElementById('time').innerText = new Date().toLocaleString();
  } catch (err) {
    document.getElementById('content').innerText = 'Error fetching state: ' + err;
    console.error(err);
  }
}

setInterval(load, 2000);
load();
</script>
</body>
</html>
"""

manager = None
state_file_path = None

def load_manager_from_file(path_str: str):
    global manager, state_file_path
    state_file_path = Path(path_str)
    if not state_file_path.exists():
        print(f"ERROR: state file not found at: {state_file_path}", file=sys.stderr)
        sys.exit(2)
    try:
        manager = ShardManager.load_state(state_file_path)
        print(f"[{time.strftime('%H:%M:%S')}] Loaded state from {state_file_path} — nodes: {len(manager.nodes)} files: {len(manager.shard_map)}")
    except Exception as e:
        print("ERROR: failed to load ShardManager state.", file=sys.stderr)
        traceback.print_exc()
        sys.exit(3)

@app.before_request
def log_request():
    print(f"[{time.strftime('%H:%M:%S')}] {request.method} {request.path} from {request.remote_addr}")

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/ping')
def ping():
    return jsonify({"ok": True, "ts": time.time()})

@app.route('/state')
def state():
    if manager is None:
        return jsonify({"error": "manager not loaded"}), 500
    try:
        data = {
            "nodes": {nid: n.to_dict() for nid, n in manager.nodes.items()},
            "shard_map": manager.shard_map,
            "file_meta": manager.file_meta,
        }
        return jsonify(data)
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "failed to serialize state"}), 500

def start_dashboard(state_file: str, host: str = "127.0.0.1", port: int = 8080):
    # Ensure manager is loaded (exit if not)
    load_manager_from_file(state_file)
    print(f"[{time.strftime('%H:%M:%S')}] Starting dashboard on http://{host}:{port} (CTRL+C to stop)")
    # Use werkzeug's built-in server (debug=False -> no auto-reloader)
    app.run(host=host, port=port, debug=False, use_reloader=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    # Force stdout/stderr to flush immediately (good for PowerShell)
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    start_dashboard(args.state_file, args.host, args.port)
