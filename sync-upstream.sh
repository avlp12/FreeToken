#!/bin/bash
# Pull upstream (FlashML-org/FreeToken) improvements into our custom branch.
# Usage: bash ~/FreeToken/sync-upstream.sh
# On conflicts: resolve, `git rebase --continue`, then re-run the push line.
set -e
cd ~/FreeToken
git fetch upstream
echo "=== upstream/main is at ==="
git log --oneline -3 upstream/main
echo "=== rebasing uran-custom onto upstream/main ==="
git checkout uran-custom
git rebase upstream/main
git push origin uran-custom --force-with-lease
echo "=== done; restart the server (Desktop START.bat) to pick up changes ==="
