#!/bin/bash
# Pull upstream (FlashML-org/FreeToken) improvements into our custom branch.
# Usage: bash ~/FreeToken/sync-upstream.sh
#
# MERGE, not rebase: uran-custom already contains a merge commit of its own
# (the pr69/pr70/pr71 stack), and rebasing across an existing merge commit
# tends to replay/duplicate/drop hunks unpredictably. A plain merge keeps
# history honest and is safe to redo if a sync is interrupted.
#
# On conflicts: resolve, `git add <file>`, `git commit` (no --continue --
# this is a merge, not a rebase), then re-run the push line.
#
# For anything nontrivial, do the integration in a separate worktree first
# (`git worktree add /root/ft-sync uran-custom`, merge upstream/main there,
# verify, then fast-forward/merge uran-custom onto the result) so this
# checkout -- which backs the live server via an editable install -- never
# sits in a conflicted state.
set -e
cd ~/FreeToken
git fetch upstream
echo "=== upstream/main is at ==="
git log --oneline -3 upstream/main
echo "=== merging upstream/main into uran-custom ==="
git checkout uran-custom
git merge upstream/main --no-edit
git push origin uran-custom
echo "=== done; restart the server (Desktop START.bat) to pick up changes ==="
