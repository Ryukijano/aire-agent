#!/usr/bin/env python3
"""AIRE Agent MCP Server — JSON-RPC over stdio.

Stdlib only. Speaks both MCP framings:
  - Content-Length headers (Cursor / TypeScript SDK)
  - NDJSON, one object per line (Claude Code / Python MCP SDK)

Dispatches generate/validate/docs tools to ~/.aire-agent/tools.
Does not wrap sbatch/squeue/scancel — those stay raw Slurm.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(REPO_DIR, "tools")
SCRIPTS_DIR = os.path.join(REPO_DIR, "scripts")
SUBPROCESS_TIMEOUT = 60
SUPPORTED_PROTOCOL_VERSIONS = (
    "2024-11-05",
    "2025-03-26",
    "2025-06-18",
)

# Set from the first inbound message so replies match the client.
_FRAMING = "ndjson"  # "headers" | "ndjson"


TOOLS = [
    {
        "name": "generate_script",
        "description": (
            "Generate a validated AIRE SBATCH job script with correct constraints, "
            "module loads, and framework boilerplate. Max 3 GPUs per node; GPU jobs "
            "need --partition=gpu and --gres=gpu:N together. Does not submit the job."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "time": {
                    "type": "string",
                    "description": "Wall time (e.g. 1h, 4h, 1d, 01:00:00). Required.",
                },
                "gpu": {
                    "type": "integer",
                    "description": "Number of GPUs (default 0 = CPU job). >3 becomes multi-node.",
                },
                "cpus": {"type": "integer", "description": "CPUs per task"},
                "mem": {
                    "type": "string",
                    "description": "Memory allocation (e.g. 8G, 85G, 200G)",
                },
                "partition": {
                    "type": "string",
                    "description": "Partition (auto: gpu if gpu>0)",
                },
                "framework": {
                    "type": "string",
                    "description": "Framework boilerplate",
                    "enum": ["pytorch", "tensorflow", "none"],
                },
                "job_name": {"type": "string", "description": "Job name (default: job)"},
                "email": {"type": "string", "description": "Notification email"},
                "array": {
                    "type": "string",
                    "description": "Array range (e.g. 1-10, 1-100%5)",
                },
            },
            "required": ["time"],
        },
    },
    {
        "name": "validate_script",
        "description": (
            "Validate an SBATCH script against AIRE rules: --time required, "
            "partition=gpu with --gres=gpu:N, max 3 GPUs per node."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "script": {
                    "type": "string",
                    "description": "Path to the job script to validate",
                },
            },
            "required": ["script"],
        },
    },
    {
        "name": "search_docs",
        "description": "Search the AIRE knowledge base and documentation for a query string.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_modules",
        "description": "List available software modules on AIRE. Optionally filter by keyword.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filter": {
                    "type": "string",
                    "description": "Optional keyword to filter modules",
                },
            },
        },
    },
    {
        "name": "system_info",
        "description": (
            "AIRE HPC hardware and storage summary (L40S GPUs, partitions, "
            "HOME vs SCRATCH). Live GPU queue is sinfo -p gpu, not this tool."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "log_experiment",
        "description": "Log an experiment run to the local JSONL experiment tracker.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Experiment name (e.g. bert_finetune_v2)",
                },
                "job_id": {
                    "type": "string",
                    "description": "Slurm job ID (defaults to $SLURM_JOB_ID)",
                },
                "metrics": {
                    "type": "string",
                    "description": 'Metrics as JSON string, e.g. \'{"loss": 0.5}\'',
                },
                "params": {
                    "type": "string",
                    "description": 'Hyperparameters as JSON string, e.g. \'{"lr": 0.001}\'',
                },
                "notes": {"type": "string", "description": "Free-text notes"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "query_experiments",
        "description": "Query logged experiments from the local tracker.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "last": {
                    "type": "integer",
                    "description": "Show last N experiments (default 20)",
                },
                "filter": {
                    "type": "string",
                    "description": "Filter by field=value",
                },
                "json_output": {
                    "type": "boolean",
                    "description": "Output raw JSONL format",
                },
            },
        },
    },
    {
        "name": "sync_docs",
        "description": "Synchronise AIRE documentation from upstream sources.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "check_quota",
        "description": (
            "Check HOME and $SCRATCH quotas on AIRE. HOME is 65 GB backed up; "
            "scratch is /mnt/scratch/$USER (Lustre). Does not run du -sh on scratch."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "doctor",
        "description": "Health-check the aire-agent toolkit install (scripts, PATH, knowledge).",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _build_args_generate_script(arguments):
    args = ["--time", arguments.get("time", "1h")]
    if arguments.get("gpu"):
        args.extend(["--gpu", str(arguments["gpu"])])
    if arguments.get("cpus"):
        args.extend(["--cpus", str(arguments["cpus"])])
    if arguments.get("mem"):
        args.extend(["--mem", arguments["mem"]])
    if arguments.get("partition"):
        args.extend(["--partition", arguments["partition"]])
    if arguments.get("framework"):
        args.extend(["--framework", arguments["framework"]])
    if arguments.get("job_name"):
        args.extend(["--job-name", arguments["job_name"]])
    if arguments.get("email"):
        args.extend(["--email", arguments["email"]])
    if arguments.get("array"):
        args.extend(["--array", arguments["array"]])
    return args


def _build_args_validate_script(arguments):
    args = []
    if arguments.get("script"):
        args.append(arguments["script"])
    return args


def _build_args_search_docs(arguments):
    args = []
    if arguments.get("query"):
        args.append(arguments["query"])
    return args


def _build_args_list_modules(arguments):
    args = []
    if arguments.get("filter"):
        args.append(arguments["filter"])
    return args


def _build_args_none(arguments):
    return []


def _build_args_log_experiment(arguments):
    args = ["--name", arguments.get("name", "unnamed")]
    if arguments.get("job_id"):
        args.extend(["--job", arguments["job_id"]])
    if arguments.get("metrics"):
        args.extend(["--metrics", arguments["metrics"]])
    if arguments.get("params"):
        args.extend(["--params", arguments["params"]])
    if arguments.get("notes"):
        args.extend(["--notes", arguments["notes"]])
    return args


def _build_args_query_experiments(arguments):
    args = []
    if arguments.get("last"):
        args.extend(["--last", str(arguments["last"])])
    if arguments.get("filter"):
        args.extend(["--filter", arguments["filter"]])
    if arguments.get("json_output"):
        args.append("--json")
    return args


TOOL_DISPATCH = {
    "generate_script": (
        os.path.join(TOOLS_DIR, "generate-script.sh"),
        _build_args_generate_script,
    ),
    "validate_script": (
        os.path.join(TOOLS_DIR, "validate-script.sh"),
        _build_args_validate_script,
    ),
    "search_docs": (os.path.join(TOOLS_DIR, "search-docs.sh"), _build_args_search_docs),
    "list_modules": (
        os.path.join(TOOLS_DIR, "list-modules.sh"),
        _build_args_list_modules,
    ),
    "system_info": (os.path.join(TOOLS_DIR, "system-info.sh"), _build_args_none),
    "log_experiment": (
        os.path.join(TOOLS_DIR, "log-experiment.sh"),
        _build_args_log_experiment,
    ),
    "query_experiments": (
        os.path.join(TOOLS_DIR, "query-experiments.sh"),
        _build_args_query_experiments,
    ),
    "sync_docs": (os.path.join(SCRIPTS_DIR, "sync.sh"), _build_args_none),
    "doctor": (os.path.join(TOOLS_DIR, "doctor.sh"), _build_args_none),
}


def _run_cmd(cmd, timeout=30):
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if result.returncode != 0 and err:
            return (out + "\n" + err).strip() if out else err
        return out or "(no output)"
    except FileNotFoundError:
        return f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return f"Timed out after {timeout}s: {' '.join(cmd)}"


def _check_quota(_arguments):
    user = os.environ.get("USER", os.environ.get("LOGNAME", ""))
    scratch = os.environ.get("SCRATCH") or f"/mnt/scratch/{user}"
    lines = [
        "=== AIRE quota ===",
        "",
        "--- HOME (backed up, 65 GB cap) ---",
        _run_cmd(["quota", "-s"]),
        "",
        f"--- SCRATCH ({scratch}) ---",
    ]
    lfs = _run_cmd(["lfs", "quota", "-hu", user, "/mnt/scratch"], timeout=45)
    if lfs.startswith("Command not found") or "not available" in lfs.lower():
        lfs = _run_cmd(["lfs", "quota", "-u", user, "/mnt/scratch"], timeout=45)
    lines.append(lfs)
    lines.extend(
        [
            "",
            "Keep HF/uv/torch/W&B caches on $SCRATCH. Source:",
            "  source /scratch/kcwp264/.aire_scratch_env.sh",
            "Do not run du -sh on Lustre scratch (it is slow and load-heavy).",
        ]
    )
    return {"error": False, "text": "\n".join(lines)}


def _system_info_overlay(text):
    extra = (
        "\n--- This account (kcwp264) ---\n"
        "  HOME:     /users/kcwp264  (65 GB backed up)\n"
        "  SCRATCH:  /mnt/scratch/kcwp264  (observed 2 TB / 1.5M files)\n"
        "  Caches:   source /scratch/kcwp264/.aire_scratch_env.sh\n"
        "  GPU jobs: --partition=gpu --gres=gpu:N (N<=3) --cpus-per-task --mem --time\n"
        "  Login A2: do not train on login-node GPUs\n"
        "  Submit:   sbatch / squeue --me / scancel / seff  (not via MCP)\n"
    )
    return (text or "") + extra


def run_tool(name, arguments):
    if name == "check_quota":
        return _check_quota(arguments)

    if name not in TOOL_DISPATCH:
        return {"error": True, "text": f"Unknown tool: {name}"}

    script_path, arg_builder = TOOL_DISPATCH[name]
    if not os.path.exists(script_path):
        return {"error": True, "text": f"Script not found: {script_path}"}

    cmd = ["bash", script_path] + arg_builder(arguments or {})
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
            cwd=REPO_DIR,
        )
        output = result.stdout
        if result.returncode != 0 and result.stderr:
            output = output + "\n" + result.stderr if output else result.stderr
        text = output.strip() if output else "(no output)"
        if name == "system_info":
            text = _system_info_overlay(text)
        return {"error": False, "text": text}
    except subprocess.TimeoutExpired:
        return {
            "error": True,
            "text": f"Tool '{name}' timed out after {SUBPROCESS_TIMEOUT}s",
        }
    except Exception as exc:
        return {"error": True, "text": f"Error running tool '{name}': {exc}"}


def handle_request(request):
    req_id = request.get("id")
    method = request.get("method", "")
    params = request.get("params") or {}

    if method == "initialize":
        client_version = params.get("protocolVersion", "2024-11-05")
        version = (
            client_version
            if client_version in SUPPORTED_PROTOCOL_VERSIONS
            else "2024-11-05"
        )
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": version,
                "capabilities": {
                    "tools": {"listChanged": False},
                    "prompts": {"listChanged": False},
                    "resources": {"listChanged": False},
                },
                "serverInfo": {
                    "name": "aire-agent",
                    "version": "1.1.0",
                },
            },
        }

    if method in ("notifications/initialized", "notifications/cancelled"):
        return None

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}

    if method == "prompts/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"prompts": []}}

    if method == "resources/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"resources": []}}

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments") or {}
        tool_result = run_tool(tool_name, arguments)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": tool_result["text"]}],
                "isError": bool(tool_result.get("error")),
            },
        }

    if req_id is None:
        return None

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def _write_response(message):
    body = json.dumps(message, ensure_ascii=False).encode("utf-8")
    if _FRAMING == "headers":
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        sys.stdout.buffer.write(header + body)
    else:
        sys.stdout.buffer.write(body + b"\n")
    sys.stdout.buffer.flush()


def _read_request():
    """Read one JSON-RPC message. Detect Content-Length vs NDJSON on first byte."""
    global _FRAMING
    raw = sys.stdin.buffer

    while True:
        ch = raw.peek(1)
        if not ch:
            return None
        if ch in (b" ", b"\t", b"\r", b"\n"):
            raw.read(1)
            continue
        break

    start = raw.peek(16)
    if not start:
        return None

    if start[:1] in (b"{", b"["):
        _FRAMING = "ndjson"
        line = raw.readline()
        if not line:
            return None
        return json.loads(line)

    _FRAMING = "headers"
    headers = {}
    while True:
        line = raw.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        decoded = line.decode("utf-8", errors="replace")
        key, _, value = decoded.partition(":")
        headers[key.strip().lower()] = value.strip()

    try:
        length = int(headers.get("content-length", "0"))
    except ValueError:
        length = 0
    body = raw.read(length) if length else b"{}"
    return json.loads(body)


def main():
    while True:
        try:
            request = _read_request()
        except json.JSONDecodeError as exc:
            _write_response(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                }
            )
            continue
        except Exception as exc:
            sys.stderr.write(f"aire-mcp read error: {exc}\n")
            sys.stderr.flush()
            return

        if request is None:
            return

        try:
            response = handle_request(request)
        except Exception as exc:
            response = {
                "jsonrpc": "2.0",
                "id": request.get("id") if isinstance(request, dict) else None,
                "error": {"code": -32603, "message": f"Internal error: {exc}"},
            }

        if response is not None:
            _write_response(response)


if __name__ == "__main__":
    main()
