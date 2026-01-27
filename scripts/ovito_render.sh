#!/usr/bin/env bash
set -e

# Ensure runtime dir exists in headless sessions
if [ -z "${XDG_RUNTIME_DIR:-}" ]; then
  export XDG_RUNTIME_DIR="/run/user/$(id -u)"
  mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"
fi

# Use EGL offscreen rendering
export QT_QPA_PLATFORM=offscreen
export QT_OPENGL=egl

exec /home/kjt/miniforge3/envs/visualization/bin/python "$@"
