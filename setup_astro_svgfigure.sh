#!/bin/bash
# =============================================================================
# setup_astro_svgfigure.sh — astro-svgfigure 生产级部署脚本
# =============================================================================
# 使用方法:
#   chmod +x setup_astro_svgfigure.sh
#   sudo bash setup_astro_svgfigure.sh              # 完整部署
#   sudo bash setup_astro_svgfigure.sh --status     # 查看状态
#   sudo bash setup_astro_svgfigure.sh --stop       # 停止服务
#   sudo bash setup_astro_svgfigure.sh --restart    # 重启服务
#   sudo bash setup_astro_svgfigure.sh --logs       # 查看日志
#   sudo bash setup_astro_svgfigure.sh --nginx      # 仅配置 Nginx
#   sudo bash setup_astro_svgfigure.sh --upgrade-node  # 升级 Node.js
#   sudo bash setup_astro_svgfigure.sh --help       # 帮助
# =============================================================================

set -e

# ─────────────────────────────────────────────────────────────────────────────
# 配置区
# ─────────────────────────────────────────────────────────────────────────────
DOMAIN="baloonet.tech"
ASTRO_DIR="/root/dylan/skynetCheapBuy/astro-svgfigure"
ASTRO_FRONTEND_PORT=4321
ASTRO_BACKEND_PORT=8000

# 复用 skynetCheapBuy 的虚拟环境
SHARED_VENV="/root/dylan/skynetCheapBuy/skynetCheapBuy/.venv"

# ─────────────────────────────────────────────────────────────────────────────
# 颜色输出
# ─────────────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

log_info()  { printf "${GREEN}[INFO]${NC} %s\n" "$1"; }
log_warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$1"; }
log_error() { printf "${RED}[ERROR]${NC} %s\n" "$1"; }
log_step()  { printf "\n${BLUE}═══ %s ═══${NC}\n" "$1"; }

# ─────────────────────────────────────────────────────────────────────────────
# 帮助信息
# ─────────────────────────────────────────────────────────────────────────────
show_help() {
    cat << 'EOF'
用法: sudo bash setup_astro_svgfigure.sh [选项]

选项:
  (无参数)        完整部署: 配置 Nginx + Systemd + 启动服务
  --status        查看所有服务状态
  --stop          停止 astro-svgfigure 服务
  --restart       重启 astro-svgfigure 服务
  --logs          查看日志
  --nginx         仅配置/更新 Nginx
  --upgrade-node  升级 Node.js 到 v22 LTS (Astro 5.x 要求 >= 18.20.8)
  --help, -h      显示帮助

EOF
}

# ─────────────────────────────────────────────────────────────────────────────
# 升级 Node.js
# ─────────────────────────────────────────────────────────────────────────────
upgrade_node() {
    log_step "升级 Node.js"
    
    CURRENT_VERSION=$(node --version 2>/dev/null || echo "未安装")
    log_info "当前版本: $CURRENT_VERSION"
    log_info "目标版本: Node.js 22 LTS"
    
    # 使用 NodeSource 安装 Node.js 22
    log_info "添加 NodeSource 仓库..."
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
    
    log_info "安装 Node.js 22..."
    apt-get install -y nodejs
    
    NEW_VERSION=$(node --version)
    log_info "升级完成: $NEW_VERSION"
    
    # 清理并重新安装项目依赖
    if [ -d "$ASTRO_DIR/node_modules" ]; then
        log_info "重新安装项目依赖..."
        cd "$ASTRO_DIR"
        rm -rf node_modules package-lock.json 2>/dev/null || true
        npm install
    fi
    
    log_info "Node.js 升级完成！现在可以运行部署脚本了"
}

# ─────────────────────────────────────────────────────────────────────────────
# 检查 Node.js 版本
# ─────────────────────────────────────────────────────────────────────────────
check_node_version() {
    if ! command -v node >/dev/null 2>&1; then
        return 1
    fi
    
    NODE_VERSION=$(node --version | sed 's/^v//')
    NODE_MAJOR=$(echo "$NODE_VERSION" | cut -d. -f1)
    NODE_MINOR=$(echo "$NODE_VERSION" | cut -d. -f2)
    NODE_PATCH=$(echo "$NODE_VERSION" | cut -d. -f3)
    
    # Astro 5.x 要求 >= 18.20.8
    if [ "$NODE_MAJOR" -lt 18 ]; then
        return 1
    elif [ "$NODE_MAJOR" -eq 18 ]; then
        if [ "$NODE_MINOR" -lt 20 ]; then
            return 1
        elif [ "$NODE_MINOR" -eq 20 ] && [ "$NODE_PATCH" -lt 8 ]; then
            return 1
        fi
    fi
    
    return 0
}

# ─────────────────────────────────────────────────────────────────────────────
# 查找 bun 路径
# ─────────────────────────────────────────────────────────────────────────────
find_bun() {
    # 尝试多个可能的位置
    for bun_path in \
        "/root/.bun/bin/bun" \
        "$HOME/.bun/bin/bun" \
        "/usr/local/bin/bun" \
        "/usr/bin/bun" \
        "$(command -v bun 2>/dev/null)"
    do
        if [ -x "$bun_path" ]; then
            echo "$bun_path"
            return 0
        fi
    done
    return 1
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. 环境检查
# ─────────────────────────────────────────────────────────────────────────────
check_environment() {
    log_step "检查环境"

    # 检查 root 权限
    if [ "$(id -u)" -ne 0 ]; then
        log_error "请使用 root 权限运行: sudo bash $0"
        exit 1
    fi

    # 检查项目目录
    if [ ! -d "$ASTRO_DIR" ]; then
        log_error "项目目录不存在: $ASTRO_DIR"
        exit 1
    fi

    # 检查 server.py
    if [ ! -f "$ASTRO_DIR/server.py" ]; then
        log_error "未找到 server.py"
        exit 1
    fi

    # 检查 Nginx
    if ! command -v nginx >/dev/null 2>&1; then
        log_error "Nginx 未安装，请先安装: apt install nginx"
        exit 1
    fi

    # 检查 bun (用 bun --bun 绕过 Node.js 版本限制)
    BUN_PATH=$(find_bun)
    if [ -z "$BUN_PATH" ]; then
        log_error "bun 未安装"
        log_info "请安装: curl -fsSL https://bun.sh/install | bash"
        exit 1
    fi
    log_info "找到 bun: $BUN_PATH"

    # 检查共享虚拟环境
    if [ ! -d "$SHARED_VENV" ]; then
        log_warn "共享虚拟环境不存在: $SHARED_VENV"
        SHARED_VENV="$ASTRO_DIR/.venv"
        if [ ! -d "$SHARED_VENV" ]; then
            log_info "创建新的虚拟环境..."
            python3 -m venv "$SHARED_VENV"
            "$SHARED_VENV/bin/pip" install -r "$ASTRO_DIR/requirements.txt"
        fi
    fi

    log_info "环境检查通过"
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. 配置 Nginx (解决冲突)
# ─────────────────────────────────────────────────────────────────────────────
configure_nginx() {
    log_step "配置 Nginx"

    # 检查 SSL 证书
    SSL_CERT="/etc/letsencrypt/live/${DOMAIN}/fullchain.pem"
    SSL_KEY="/etc/letsencrypt/live/${DOMAIN}/privkey.pem"

    if [ -f "$SSL_CERT" ] && [ -f "$SSL_KEY" ]; then
        HAS_SSL=true
        log_info "检测到 Let's Encrypt 证书"
    else
        HAS_SSL=false
        log_warn "未检测到 SSL 证书，将使用 HTTP"
    fi

    # ═══════════════════════════════════════════════════════════════════════
    # 关键: 禁用冲突的 Nginx 配置
    # ═══════════════════════════════════════════════════════════════════════
    log_info "解决 Nginx 配置冲突..."
    
    for conf in default skynetCheapBuy skynet chatbot; do
        if [ -f "/etc/nginx/sites-enabled/$conf" ]; then
            rm -f "/etc/nginx/sites-enabled/$conf"
            log_info "已禁用冲突配置: $conf"
        fi
    done

    # 备份现有配置
    if [ -f /etc/nginx/sites-available/astro-svgfigure ]; then
        cp /etc/nginx/sites-available/astro-svgfigure \
           /etc/nginx/sites-available/astro-svgfigure.bak.$(date +%Y%m%d%H%M%S)
    fi

    # 创建 Nginx 配置
    if [ "$HAS_SSL" = true ]; then
        cat > /etc/nginx/sites-available/astro-svgfigure << NGINX_CONF
# astro-svgfigure Nginx 配置 (HTTPS)
# 生成时间: $(date '+%Y-%m-%d %H:%M:%S')

server {
    listen 80;
    listen [::]:80;
    server_name ${DOMAIN} www.${DOMAIN};

    location ^~ /.well-known/acme-challenge/ {
        root /var/www/html;
        allow all;
    }

    location / {
        return 301 https://\$host\$request_uri;
    }
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name ${DOMAIN} www.${DOMAIN};

    ssl_certificate ${SSL_CERT};
    ssl_certificate_key ${SSL_KEY};
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;

    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;

    # ============================================================
    # API 路由规则（顺序重要！）
    # ============================================================
    
    # Python 后端 API - 只转发特定的后端路径
    # topology, beautify, validate, export, models, config, generate-prompt, run, artifacts, animation
    location ~ ^/api/(topology|beautify|validate|export|models|config|generate-prompt|run|artifacts|animation) {
        proxy_pass http://127.0.0.1:${ASTRO_BACKEND_PORT};
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_read_timeout 600s;
        proxy_connect_timeout 75s;
        proxy_send_timeout 600s;
        client_max_body_size 50M;
        proxy_buffering off;
    }

    # Astro API（包括 /api/health）- 转发给前端
    location /api/ {
        proxy_pass http://127.0.0.1:${ASTRO_FRONTEND_PORT};
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_read_timeout 300s;
    }

    # 前端 Astro
    location / {
        proxy_pass http://127.0.0.1:${ASTRO_FRONTEND_PORT};
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_cache_bypass \$http_upgrade;
        proxy_read_timeout 300s;
    }
}
NGINX_CONF
    else
        cat > /etc/nginx/sites-available/astro-svgfigure << NGINX_CONF
# astro-svgfigure Nginx 配置 (HTTP)
server {
    listen 80;
    listen [::]:80;
    server_name ${DOMAIN} www.${DOMAIN};

    location ^~ /.well-known/acme-challenge/ {
        root /var/www/html;
        allow all;
    }

    # Python 后端 API - 只转发特定的后端路径
    location ~ ^/api/(topology|beautify|validate|export|models|config|generate-prompt|run|artifacts|animation) {
        proxy_pass http://127.0.0.1:${ASTRO_BACKEND_PORT};
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_read_timeout 600s;
        client_max_body_size 50M;
        proxy_buffering off;
    }

    # Astro API（包括 /api/health）
    location /api/ {
        proxy_pass http://127.0.0.1:${ASTRO_FRONTEND_PORT};
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_read_timeout 300s;
    }

    location / {
        proxy_pass http://127.0.0.1:${ASTRO_FRONTEND_PORT};
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
        proxy_read_timeout 300s;
    }
}
NGINX_CONF
    fi

    # 启用站点
    ln -sf /etc/nginx/sites-available/astro-svgfigure /etc/nginx/sites-enabled/

    # 测试配置
    log_info "测试 Nginx 配置..."
    if nginx -t 2>&1; then
        systemctl reload nginx
        log_info "Nginx 配置完成"
    else
        log_error "Nginx 配置错误"
        nginx -t
        exit 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. 创建 Systemd 服务 (使用 bun)
# ─────────────────────────────────────────────────────────────────────────────
create_systemd_services() {
    log_step "创建 Systemd 服务"
    
    # Python 路径
    if [ -f "$SHARED_VENV/bin/python" ]; then
        PYTHON_BIN="$SHARED_VENV/bin/python"
    else
        PYTHON_BIN="$ASTRO_DIR/.venv/bin/python"
    fi

    # bun 路径
    BUN_PATH=$(find_bun)
    if [ -z "$BUN_PATH" ]; then
        log_error "bun 未找到"
        exit 1
    fi

    log_info "使用 bun: $BUN_PATH"
    log_info "使用 Python: $PYTHON_BIN"

    # 后端服务 (Python FastAPI) - 加载 .env 环境变量
    cat > /etc/systemd/system/astro-backend.service << EOF
[Unit]
Description=Astro SVGFigure Python Backend (Port ${ASTRO_BACKEND_PORT})
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=${ASTRO_DIR}
# 加载 .env 文件
EnvironmentFile=-${ASTRO_DIR}/.env
ExecStart=${PYTHON_BIN} server.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
# 增加启动超时
TimeoutStartSec=60

[Install]
WantedBy=multi-user.target
EOF

    # 前端服务 - 直接用 bun 运行（绕过 Node.js 版本检查）
    cat > /etc/systemd/system/astro-frontend.service << EOF
[Unit]
Description=Astro SVGFigure Frontend (Port ${ASTRO_FRONTEND_PORT})
After=network.target astro-backend.service
# 等待后端启动
Requires=astro-backend.service

[Service]
Type=simple
User=root
WorkingDirectory=${ASTRO_DIR}
# 设置 PATH 确保能找到 bun
Environment=PATH=/root/.bun/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=${BUN_PATH} run dev --host 0.0.0.0
Restart=always
RestartSec=5
Environment=HOST=0.0.0.0
Environment=PORT=${ASTRO_FRONTEND_PORT}
# 增加启动超时（Astro 首次启动较慢）
TimeoutStartSec=120

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable astro-backend astro-frontend

    log_info "Systemd 服务已创建"
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. 安装依赖
# ─────────────────────────────────────────────────────────────────────────────
install_dependencies() {
    log_step "检查依赖"

    cd "$ASTRO_DIR"
    BUN_PATH=$(find_bun)

    # Node.js 依赖
    if [ ! -d "node_modules" ]; then
        log_info "安装依赖 (bun install)..."
        "$BUN_PATH" install
    else
        log_info "node_modules 已存在"
    fi

    # Python 依赖
    if [ -f "$SHARED_VENV/bin/pip" ]; then
        log_info "检查 Python 依赖..."
        "$SHARED_VENV/bin/pip" install -q -r requirements.txt 2>/dev/null || true
    fi

    log_info "依赖检查完成"
}

# ─────────────────────────────────────────────────────────────────────────────
# 5. 启动/停止服务
# ─────────────────────────────────────────────────────────────────────────────
stop_services_quiet() {
    systemctl stop astro-frontend 2>/dev/null || true
    systemctl stop astro-backend 2>/dev/null || true
    pkill -f "bun.*astro" 2>/dev/null || true
    pkill -f "astro dev" 2>/dev/null || true
    sleep 1
}

start_services() {
    log_step "启动服务"

    stop_services_quiet

    log_info "启动后端服务..."
    systemctl start astro-backend
    sleep 3

    if systemctl is-active --quiet astro-backend; then
        log_info "后端服务启动成功"
    else
        log_error "后端服务启动失败"
        journalctl -u astro-backend --no-pager -n 10
        exit 1
    fi

    log_info "启动前端服务..."
    systemctl start astro-frontend
    
    # Astro 首次启动需要较长时间（约10-15秒）
    log_info "等待 Astro 启动 (约15秒)..."
    
    # 等待端口监听，最多等30秒
    for i in 1 2 3 4 5 6; do
        sleep 5
        if ss -tlnp 2>/dev/null | grep -q ":${ASTRO_FRONTEND_PORT} "; then
            log_info "前端服务启动成功 (端口 ${ASTRO_FRONTEND_PORT} 已监听)"
            return 0
        fi
        log_info "等待中... ($((i*5))秒)"
    done

    # 检查服务状态
    if systemctl is-active --quiet astro-frontend; then
        log_warn "服务运行中但端口未监听，查看日志:"
        journalctl -u astro-frontend --no-pager -n 15
    else
        log_error "前端服务启动失败"
        journalctl -u astro-frontend --no-pager -n 15
        exit 1
    fi
}

stop_services() {
    log_step "停止服务"
    stop_services_quiet
    log_info "服务已停止"
}

restart_services() {
    log_step "重启服务"
    stop_services_quiet
    sleep 2
    start_services
}

# ─────────────────────────────────────────────────────────────────────────────
# 6. 显示状态
# ─────────────────────────────────────────────────────────────────────────────
show_status() {
    log_step "服务状态"

    echo ""

    if systemctl is-active --quiet astro-frontend; then
        log_info "✅ 前端服务: 运行中 (端口 ${ASTRO_FRONTEND_PORT})"
    else
        log_warn "⚠️  前端服务: 未运行"
    fi

    if systemctl is-active --quiet astro-backend; then
        log_info "✅ 后端服务: 运行中 (端口 ${ASTRO_BACKEND_PORT})"
    else
        log_warn "⚠️  后端服务: 未运行"
    fi

    if systemctl is-active --quiet nginx; then
        log_info "✅ Nginx: 运行中"
    else
        log_warn "⚠️  Nginx: 未运行"
    fi

    echo ""
    for port in ${ASTRO_FRONTEND_PORT} ${ASTRO_BACKEND_PORT}; do
        if ss -tlnp 2>/dev/null | grep -q ":${port} "; then
            log_info "✅ 端口 ${port}: 已监听"
        else
            log_warn "⚠️  端口 ${port}: 未监听"
        fi
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# 7. 显示日志
# ─────────────────────────────────────────────────────────────────────────────
show_logs() {
    log_step "日志"
    echo "前端: journalctl -u astro-frontend -f"
    echo "后端: journalctl -u astro-backend -f"
    echo ""
    log_info "最近后端日志:"
    journalctl -u astro-backend --no-pager -n 15 2>/dev/null || echo "(无)"
    echo ""
    log_info "最近前端日志:"
    journalctl -u astro-frontend --no-pager -n 15 2>/dev/null || echo "(无)"
}

# ─────────────────────────────────────────────────────────────────────────────
# 8. 部署完成信息
# ─────────────────────────────────────────────────────────────────────────────
show_deployment_info() {
    if [ -f "/etc/letsencrypt/live/${DOMAIN}/fullchain.pem" ]; then
        PROTOCOL="https"
    else
        PROTOCOL="http"
    fi

    echo ""
    printf "${GREEN}══════════════════════════════════════${NC}\n"
    printf "${GREEN}  astro-svgfigure 部署完成 🎉${NC}\n"
    printf "${GREEN}══════════════════════════════════════${NC}\n"
    echo ""
    echo "访问地址:"
    echo "  主页:     ${PROTOCOL}://${DOMAIN}/"
    echo "  生成页:   ${PROTOCOL}://${DOMAIN}/generate"
    echo ""
    echo "管理命令:"
    echo "  sudo bash $0 --restart"
    echo "  sudo bash $0 --status"
    echo "  sudo bash $0 --logs"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────
main() {
    case "${1:-}" in
        --help|-h)
            show_help
            ;;
        --status)
            show_status
            ;;
        --stop)
            check_environment
            stop_services
            ;;
        --restart)
            check_environment
            restart_services
            show_status
            ;;
        --logs)
            show_logs
            ;;
        --nginx)
            check_environment
            configure_nginx
            ;;
        --upgrade-node)
            if [ "$(id -u)" -ne 0 ]; then
                log_error "请使用 root 权限运行"
                exit 1
            fi
            upgrade_node
            ;;
        *)
            log_step "开始部署 astro-svgfigure"
            echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"

            check_environment
            install_dependencies
            create_systemd_services
            configure_nginx
            start_services
            show_status
            show_deployment_info
            ;;
    esac
}

main "$@"