#!/data/data/com.termux/files/usr/bin/bash
# Auto-start the bot on phone boot (Termux:Boot).
#
#   mkdir -p ~/.termux/boot
#   cp scripts/termux-boot.sh ~/.termux/boot/
#
# The app must be installed with Termux:Boot and battery optimisation disabled
# for Termux, or Android will kill the process.
set -u
cd "$(dirname "$0")/../.." 2>/dev/null || cd ~/forex-candle-predictor
termux-wake-lock
set -a
[ -f .env ] && . ./.env
set +a
exec python bot.py >> ~/bot.log 2>&1
