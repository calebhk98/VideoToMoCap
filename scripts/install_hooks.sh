#!/usr/bin/env bash
# Point git at the repo's tracked hooks so the code-health check runs on commit.
# Run once after cloning:  bash scripts/install_hooks.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
chmod +x .githooks/pre-commit scripts/check_code_health.py 2>/dev/null || true
git config core.hooksPath .githooks
echo "Installed: git will now run .githooks/pre-commit (code-health check) on every commit."
echo "Bypass a single commit with 'git commit --no-verify' if you must."
