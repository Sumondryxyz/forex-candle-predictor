#!/data/data/com.termux/files/usr/bin/bash
# One-shot Termux setup for the Quotex next-candle predictor bot.
#
#   bash scripts/install-termux.sh
#
# Installs Python, the Python dependencies and (optionally) pyquotex, then writes
# a .env file you can edit. Run it from the repository root.
set -u

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

say "Updating packages"
pkg update -y && pkg upgrade -y

say "Installing Python and build tools"
pkg install -y python git clang libxml2 libxslt openssl rust binutils

say "Installing Python dependencies"
pip install --upgrade pip
pip install -r requirements.txt

read -r -p "Install pyquotex for the Quotex source? [y/N] " q
case "$q" in
  [yY]*) pip install "git+https://github.com/cleitonleonel/pyquotex.git" ;;
  *) echo "Skipping pyquotex (TradingView source still works)." ;;
esac

if [ ! -f .env ]; then
  say "Writing .env (edit it before running the bot)"
  cat > .env <<'ENV'
# Telegram: talk to @BotFather, /newbot, paste the token here.
TELEGRAM_BOT_TOKEN=

# Data source: tradingview needs no login; quotex needs the credentials below.
DATA_SOURCE=tradingview

# Only for DATA_SOURCE=quotex. Use a PRACTICE account while testing.
QUOTEX_EMAIL=
QUOTEX_PASSWORD=
QUOTEX_ACCOUNT=PRACTICE

# Signals below this confidence are labelled "below floor" and not pushed.
MIN_CONF=0.70
WEB_PAIRS=EURUSD,GBPUSD,USDJPY,AUDUSD,USDCHF,USDCAD
WEB_INTERVALS=1m,5m,15m

# Auto-push check interval in seconds.
PUSH_INTERVAL=30
ENV
else
  echo ".env already exists; leaving it alone."
fi

say "Verifying"
python -c "import predictor.telegram_bot" && echo "import OK"
python -m pytest -q tests/test_telegram_bot.py || true

cat <<'NEXT'

Done. Next:
  1. Edit .env and set TELEGRAM_BOT_TOKEN (and Quotex creds if you use them).
  2. Load it and run the bot:
       set -a; . ./.env; set +a
       python bot.py
  3. Message your bot in Telegram and send /start.

To keep it alive: install the Termux:Boot app, then add
  scripts/termux-boot.sh
to ~/.termux/boot/ and disable battery optimisation for Termux.
NEXT
