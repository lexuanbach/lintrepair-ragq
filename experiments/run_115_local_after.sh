#!/bin/bash
cd /Users/xuanbachle/Desktop/05_Notes/2026/Quantum/P4-lintrepair-ragq
source .venv/bin/activate
PID=$(cat experiments/results/new_local_models.pid 2>/dev/null)
echo "[$(date)] waiting for Q-Defects40 run (pid $PID) to finish before 115-bug crossover..."
while [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "[$(date)] Q-Defects40 done -> starting 115-bug powered crossover on local open-weights"
python3 experiments/run_mutation_crossover.py "qwen2.5:7b" "qwen2.5:14b" "llama3.3:70b" "qwen2.5:72b"
echo "[$(date)] 115-bug local crossover DONE"
