#!/bin/bash
cd /workspaces/stock-scout-2
git add stock_scout.py
git commit -m "fix: correct indentation errors in technical indicators loop

- Fixed continue statement indentation in line 1889
- Fixed MACD/ADX block indentation
- Fixed all subsequent calculations indentation
- All code now properly aligned at for-loop level"
git push origin main
echo "Done!"
