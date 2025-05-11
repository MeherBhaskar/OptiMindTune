#!/usr/bin/env bash
set -e

# Base directory
BASE_DIR="OptiMindTune/agents"

# List of agents and their subagents
declare -A AGENTS=(
  ["model_selector"]="data_profiler model_heuristic model_ranker"
  ["data_inspector"]=""
  ["trainer"]=""
)

# Create base agents folder
mkdir -p "${BASE_DIR}"

# Loop through each top-level agent
for AGENT in "${!AGENTS[@]}"; do
  AGENT_DIR="${BASE_DIR}/${AGENT}"
  SUBAGENTS=${AGENTS[$AGENT]}

  # Create agent folder and files
  mkdir -p "${AGENT_DIR}"
  touch "${AGENT_DIR}/__init__.py"
  touch "${AGENT_DIR}/agent.py"

  # If there are subagents, scaffold them
  if [ -n "${SUBAGENTS}" ]; then
    SUB_DIR="${AGENT_DIR}/subagents"
    mkdir -p "${SUB_DIR}"
    touch "${SUB_DIR}/__init__.py"

    for SA in ${SUBAGENTS}; do
      SA_DIR="${SUB_DIR}/${SA}"
      mkdir -p "${SA_DIR}"
      touch "${SA_DIR}/__init__.py"
      touch "${SA_DIR}/agent.py"
    done
  fi
done

echo "Scaffolding complete under ${BASE_DIR}"
