# Sourced (after scripts/env.sh): run a general engine from its own Docker image in a machine that has no docker.
#   img_pull NAME REPO@sha256:DIGEST     unpack the image (this host's architecture) into $MK_ROOT/img/NAME
#   img_exec NAME 'shell command'        run the command in the image's environment
# Two ways, the first that works: (1) chroot into the unpacked tree with the node's /dev, /proc, /sys, /tmp and
# $MK_ROOT bound in (needs a machine that may mount); (2) "transplant": the image's own Python and libraries run on the
# host kernel and C library (its glibc is older than the host's), with the image's install prefixes (/opt/...,
# /usr/local/lib/python3.*) reachable by symlink. Which one was used is printed (MK_ENV img_mode).

IMG_ARCH="$(uname -m | sed 's/x86_64/amd64/; s/aarch64/arm64/')"

img_pull() {  # NAME REF
  local name="$1" ref="$2" R="$MK_ROOT/img/$1" t=$(date +%s)
  [ -x "$R/bin/bash" ] || python3 "$MK_REPO/scripts/ink/oci_pull.py" "$ref" "$R" --arch "$IMG_ARCH" 2>&1 | grep -v image_env | tail -4
  [ -x "$R/bin/bash" ] || { stage "image $name" failed $t; return 1; }
  echo "$ref" > "$MK_ROOT/img/$name.ref"
  echo "MK_ENV img $name ref=$ref arch=$IMG_ARCH size=$(du -sh "$R" 2>/dev/null | cut -f1)"
  stage "image $name" ok $t
}

_img_chroot_ready() {  # NAME
  local R="$MK_ROOT/img/$1"
  [ -f "$R/.mk_chroot" ] && return 0
  local d
  for d in dev proc sys tmp; do mkdir -p "$R/$d"; mount --rbind "/$d" "$R/$d" 2>/dev/null || return 1; done
  mkdir -p "$R$MK_ROOT"; mountpoint -q "$R$MK_ROOT" || mount --rbind "$MK_ROOT" "$R$MK_ROOT" 2>/dev/null || return 1
  mkdir -p "$R$MK_REPO"; mountpoint -q "$R$MK_REPO" || mount --rbind "$MK_REPO" "$R$MK_REPO" 2>/dev/null || true
  cp /etc/resolv.conf "$R/etc/resolv.conf" 2>/dev/null; cp /etc/hosts "$R/etc/hosts" 2>/dev/null
  chroot "$R" /bin/true || return 1
  touch "$R/.mk_chroot"
}

_img_transplant_env() {  # NAME: prints the shell lines (each ending in ";") that set up the transplant mode
  local R="$MK_ROOT/img/$1" farm="$MK_ROOT/img/$1.libs" d venv="" home=""
  if [ ! -d "$farm" ]; then
    mkdir -p "$farm"
    # every shared library of the image except the C library itself (the host's is newer and must be the one used)
    for d in "$R/usr/lib/$(uname -m)-linux-gnu" "$R/lib/$(uname -m)-linux-gnu" "$R/usr/local/cuda/lib64" "$R/usr/local/lib"; do
      [ -d "$d" ] || continue
      find "$d" -maxdepth 1 -name '*.so*' 2>/dev/null | while read -r f; do
        case "$(basename "$f")" in ld-linux*|libc.so*|libm.so*|libpthread*|libdl.so*|librt.so*|libresolv*|libutil.so*|libnss_*|libmvec*|libanl*|libBrokenLocale*|libthread_db*|libc_malloc*|libmemusage*|libpcprofile*) continue ;; esac
        [ -e "$farm/$(basename "$f")" ] || ln -s "$f" "$farm/$(basename "$f")"
      done
    done
    # absolute prefixes the image's files refer to (venv and its interpreter, shebangs, CUDA_HOME, workspaces)
    for d in /opt/sglang /opt/python /opt/venv /sgl-workspace /vllm-workspace /usr/local/lib/python3.12; do
      [ -e "$R$d" ] && [ ! -e "$d" ] && { mkdir -p "$(dirname "$d")"; ln -s "$R$d" "$d"; }
    done
  fi
  for d in "$R"/opt/*/pyvenv.cfg; do [ -f "$d" ] && venv="$(dirname "$d")" && break; done
  if [ -n "$venv" ]; then  # a venv on a relocatable interpreter (uv): runs as it is through the symlinked prefixes
    echo "export MK_IMG_PY='/opt/$(basename "$venv")/bin/python3';"
  else  # the distribution's interpreter: its own stdlib through PYTHONHOME, its dist-packages through PYTHONPATH
    echo "export PYTHONHOME='$R/usr' PYTHONPATH='$R/usr/local/lib/python3.12/dist-packages:$R/usr/lib/python3/dist-packages' MK_IMG_PY='$R/usr/bin/python3.12';"
  fi
  echo "export LD_LIBRARY_PATH='$farm:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}';"
  echo "export PATH=\"\$(dirname \"\$MK_IMG_PY\"):$R/usr/local/cuda/bin:$R/usr/local/bin:\$PATH\" CUDA_HOME='$R/usr/local/cuda';"
  echo "python3() { \"\$MK_IMG_PY\" \"\$@\"; }; python() { \"\$MK_IMG_PY\" \"\$@\"; }; export -f python3 python;"
}

img_mode() {  # NAME -> chroot | transplant
  if [ -z "${IMG_MODE:-}" ]; then
    if _img_chroot_ready "$1"; then IMG_MODE=chroot; else IMG_MODE=transplant; fi
    echo "MK_ENV img_mode=$IMG_MODE" >&2
  fi
  echo "$IMG_MODE"
}

img_exec() {  # NAME 'command'
  local name="$1" cmd="$2" R="$MK_ROOT/img/$1"
  local cfg="$MK_ROOT/img/$1.config.json"
  # the image's own environment (PATH, LD_LIBRARY_PATH, its settings), then ours on top
  local imgenv; imgenv="$(python3 -c 'import json,sys,shlex; e=json.load(open(sys.argv[1]))["config"].get("Env",[]); print(" ".join("export "+shlex.quote(x)+";" for x in e))' "$cfg")"
  local ours="export MK_PORT='$MK_PORT' MK_ROOT='$MK_ROOT' HF_HOME='$HF_HOME' HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONUNBUFFERED=1 HOME=/root;"
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && ours="$ours export CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES';"
  [ -n "${IMG_EXTRA_ENV:-}" ] && ours="$ours export $IMG_EXTRA_ENV;"
  if [ "$(img_mode "$name")" = chroot ]; then
    chroot "$R" /usr/bin/env -i /bin/bash -c "$imgenv $ours cd /; $cmd"
  else
    bash -c "$imgenv
$(_img_transplant_env "$name")
$ours
$cmd"
  fi
}
