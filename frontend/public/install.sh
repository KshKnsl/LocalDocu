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

  info "Using $PY_CMD — installing Python deps, ensuring Ollama and downloading Hindices.py"
  $PY_CMD -m pip install --upgrade pip setuptools wheel || true
  $PY_CMD -m pip install fastapi uvicorn pyngrok requests boto3 python-multipart aiofiles langchain langchain-community chromadb sentence-transformers PyMuPDF langchain-huggingface langchain-chroma langchain-ollama langchain-experimental flashrank pydantic python-dotenv || true
  if ! command -v ollama >/dev/null 2>&1; then
    info "Installing Ollama (official script)"
    if curl -fsSL https://ollama.com/install.sh | sh; then
      info "Ollama installed"
    else
      err "Ollama install failed or requires manual intervention"
    fi
  fi
  if command -v ollama >/dev/null 2>&1; then
    ollama pull gemma3:1b || true
    ollama pull llava || true
  fi

  # download the latest Hindices.py for manual run
  curl -fsSL https://raw.githubusercontent.com/KshKnsl/LocalDocu/main/ai-backend/Hindices.py -o "$INSTALL_DIR/Hindices.py"

  BIN_DIR="$HOME/.local/bin"
  mkdir -p "$BIN_DIR"
  cat > "$BIN_DIR/localdocu-run" <<'EOF'
#!/usr/bin/env bash
# localdocu-run — starts the Hindices.py backend
exec python "$HOME/.localdocu-backend/Hindices.py" "$@"
EOF
  chmod +x "$BIN_DIR/localdocu-run"
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
  echo "  localdocu-run"
  echo "  (or: python $INSTALL_DIR/Hindices.py)"
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
    echo "  localdocu-run"
    echo "  (or: python $INSTALL_DIR/Hindices.py)"
    echo
    exit 0
  fi
fi

err "Automatic setup failed. Please install Python 3.10+, then run the installer again or follow manual instructions in the README."
exit 1
