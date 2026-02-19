set -euo pipefail

print() { printf '%s\n' "$*"; }
info() { print "[info] $*"; }
succ() { print "[ok] $*"; }
err() { print "[error] $*" >&2; }

INSTALL_DIR="$HOME/.localdocu-backend"
mkdir -p "$INSTALL_DIR"
info "install directory: $INSTALL_DIR"

run_bootstrap_direct() {
  PY_CMD=""
  if command -v python >/dev/null 2>&1; then
    PY_CMD=python
  else
    return 1
  fi

  info "Using $PY_CMD - creating an isolated virtual environment and installing pinned Python dependencies"
  VENV_DIR="$INSTALL_DIR/venv"
  if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment at $VENV_DIR"
    if ! $PY_CMD -m venv "$VENV_DIR"; then
      info "python -m venv failed - trying virtualenv via pip"
      $PY_CMD -m pip install --user virtualenv || true
      $PY_CMD -m virtualenv "$VENV_DIR" || err "Failed to create virtualenv"
    fi
  fi
  PY_VENV="$VENV_DIR/bin/python"
  info "Upgrading pip, setuptools, wheel in the virtualenv..."
  "$PY_VENV" -m pip install --upgrade pip setuptools wheel

  info "Installing pinned Python packages from requirements.txt"
  curl -fsSL https://raw.githubusercontent.com/KshKnsl/LocalDocu/main/ai-backend/requirements.txt -o "$INSTALL_DIR/requirements.txt" || err "Failed to download requirements.txt"
  "$PY_VENV" -m pip install -r "$INSTALL_DIR/requirements.txt" || err "pip install -r requirements.txt failed"

  info "Checking for Ollama"
  if ! command -v ollama >/dev/null 2>&1; then
    info "Installing Ollama (official script)"
    if curl -fL https://ollama.com/install.sh | sh; then
      succ "Ollama installer finished"
    else
      err "Ollama install failed or requires manual intervention"
    fi
  fi
  if command -v ollama >/dev/null 2>&1; then
    info "Pulling Ollama models (gemma3:1b, llava) - this may take a while"
    ollama pull gemma3:1b || err "gemma3 pull failed (ok to ignore)"
    ollama pull llava || err "llava pull failed (ok to ignore)"
  fi

  # download the latest Hindices.py for manual run
  info "Downloading Hindices.py to $INSTALL_DIR/Hindices.py"
  curl -fL https://raw.githubusercontent.com/KshKnsl/LocalDocu/main/ai-backend/Hindices.py -o "$INSTALL_DIR/Hindices.py" --progress-bar || err "Failed to download Hindices.py"
  succ "Downloaded Hindices.py"

  info "Creating localdocu-run shim in $HOME/.local/bin" 
  BIN_DIR="$HOME/.local/bin"
  mkdir -p "$BIN_DIR"
  cat > "$BIN_DIR/localdocu-run" <<'EOF'
#!/usr/bin/env bash
# localdocu-run - starts the Hindices.py backend (uses isolated venv)
exec "$HOME/.localdocu-backend/venv/bin/python" "$HOME/.localdocu-backend/Hindices.py" "$@"
EOF
  chmod +x "$BIN_DIR/localdocu-run"
  succ "Created shim: $BIN_DIR/localdocu-run"

  if ! echo ":$PATH:" | grep -q ":$BIN_DIR:"; then
    PROFILE_FILE="$HOME/.profile"
    if [ -n "${ZSH_VERSION-}" ] && [ -f "$HOME/.zshrc" ]; then
      PROFILE_FILE="$HOME/.zshrc"
    elif [ -n "${BASH_VERSION-}" ] && [ -f "$HOME/.bashrc" ]; then
      PROFILE_FILE="$HOME/.bashrc"
    fi
    if ! grep -Fq 'export PATH="$HOME/.local/bin:$PATH"' "$PROFILE_FILE" 2>/dev/null; then
      printf '\n# LocalDocu: add local bin to PATH\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$PROFILE_FILE"
      info "Added $BIN_DIR to PATH in $PROFILE_FILE (restart your shell or run 'source $PROFILE_FILE')"
    fi
  fi



  return 0
}

try_install_python() {
  if command -v brew >/dev/null 2>&1; then
    info "Homebrew found — installing python@3"
    brew update || true
    brew install python || return 1
    return 0
  fi

  if command -v apt-get >/dev/null 2>&1; then
    info "apt-get found — installing system Python packages"
    sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip unzip curl || return 1
    return 0
  fi
  return 1
}

if run_bootstrap_direct; then
  succ "Setup completed — dependencies installed and Hindices.py downloaded"
  echo
  echo "Start the backend manually with:"
  echo "  localdocu-run  (uses isolated venv at $INSTALL_DIR/venv)"
  echo "  (or: $INSTALL_DIR/venv/bin/python $INSTALL_DIR/Hindices.py)"
  echo
  exit 0
fi

info "Python not available — attempting to install via package manager"

if try_install_python; then
  sleep 2
  if run_bootstrap_direct; then
    succ "Setup completed after installing Python"
    echo
    echo "Start the backend manually with:"
    echo "  localdocu-run  (uses isolated venv at $INSTALL_DIR/venv)"
    echo "  (or: $INSTALL_DIR/venv/bin/python $INSTALL_DIR/Hindices.py)"
    echo
    exit 0
  fi
fi

err "Automatic setup failed. Please install Python 3.10+, then run the installer again or follow manual instructions in the README."
exit 1
