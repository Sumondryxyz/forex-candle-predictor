#!/data/data/com.termux/files/usr/bin/bash
# One-shot Termux setup for the Quotex next-candle predictor bot.
#
#   bash scripts/install-termux.sh
#
# Installs Python and the light dependencies, writes a .env, then optionally
# installs the heavy model extras (pandas/scikit-learn) and pyquotex. Run it
# from the repository root.
set -u

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }
ask() {
  # ask "<question>" -> returns 0 for yes
  local q="$1" a
  read -r -p "$q [y/N] " a
  case "$a" in [yY]*) return 0 ;; *) return 1 ;; esac
}

say "Updating packages"
pkg update -y && pkg upgrade -y

say "Installing Python and build tools"
pkg install -y python git clang libxml2 libxslt openssl binutils

say "Installing the light dependencies (the bot runs on these alone)"
pip install --upgrade pip wheel
pip install "numpy>=1.24" "requests>=2.31" "flask>=3.0" "websockets>=13.0"

if ask "Install pandas + scikit-learn? Richer features, slower install on phone"; then
  pip install "pandas>=2.0" "scikit-learn>=1.3"
fi

if ask "Install pyquotex for the Quotex source? Builds curl-cffi and needs Rust (slow on phone)"; then
  pkg install -y rust
  pip install "websockets>=12.0" "beautifulsoup4>=4.12" "curl-cffi>=0.15" \
              "fake-useragent==2.2.0" "pyfiglet>=1.0" "rich>=13.7" "orjson>=3.9"
  pip install "git+https://github.com/cleitonleonel/pyquotex.git"
else
  echo "Skipping pyquotex. The TradingView source needs no login and works now."
fi

if [ ! -f .env ]; then
  say "Writing .env (edit it before running the bot)"
  cat > .env <<'ENV'
# Telegram: talk to @BotFather on Telegram, /newbot, paste the token here.
TELEGRAM_BOT_TOKEN=

# Data source: tradingview needs no login; quotex needs credentials below.
DATA_SOURCE=tradingview

# Only for DATA_SOURCE=quotex. Use a PRACTICE account while testing.
QUOTEX_EMAIL=
QUOTEX_PASSWORD=
QUOTEX_ACCOUNT=PRACTICE
QUOTEX_SESSION_FILE=

# A signal below this confidence is labelled "below floor" and not pushed.
MIN_CONF=0.70
WEB_PAIRS=EURUSD,GBPUSD,USDJPY,AUDUSD,USDCHF,USDCAD
WEB_INTERVALS=1m,5m,15m
PUSH_INTERVAL=30
ENV
else
  echo ".env already exists; leaving it alone."
fi

say "Verifying the bot imports"
if python -c "import predictor.telegram_bot" 2>/dev/null; then
  echo "import OK"
else
  echo "Import failed. Re-run and check the errors above."
fi

cat <<'NEXT'

Done. Next:
  1. Edit .env and set TELEGRAM_BOT_TOKEN (nano .env).
  2. Load it and run the bot:
       set -a; . ./.env; set +a
       python bot.py
  3. In Telegram, open your bot and send /start.

To keep it running after a reboot install the Termux:Boot app, then
  mkdir -p ~/.termux/boot
  cp scripts/termux-boot.sh ~/.termux/boot/
and disable battery optimisation for Termux.
NEXT
