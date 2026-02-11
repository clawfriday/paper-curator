#!/bin/bash
#
# HPC Services Manager for Paper Curator
# Manages Singularity containers for frontend, backend, and database
#
# Usage:
#   ./scripts/hpc-services.sh start|stop|status|logs [service]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SIF_DIR="${PROJECT_ROOT}/containers"
LOG_DIR="${PROJECT_ROOT}/logs"
RUN_DIR="${PROJECT_ROOT}/run"
CONFIG_FILE="${PROJECT_ROOT}/config/config.yaml"

# Function to read value from config.yaml
# Usage: read_config "key_name" "default_value"
read_config() {
    local key="$1"
    local default="$2"
    local value
    
    if [[ -f "$CONFIG_FILE" ]]; then
        # Simple YAML parser using grep and awk
        value=$(grep -E "^\s*${key}:" "$CONFIG_FILE" 2>/dev/null | head -1 | awk -F: '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')
    fi
    
    echo "${value:-$default}"
}

# Read ports from config/config.yaml (with fallback defaults)
BACKEND_PORT=$(read_config "backend_port" "3100")
FRONTEND_PORT=$(read_config "frontend_port" "3000")
DB_PORT="${DB_PORT:-5432}"

# Database settings
DB_USER="${POSTGRES_USER:-curator}"
DB_PASS="${POSTGRES_PASSWORD:-curator}"
DB_NAME="${POSTGRES_DB:-paper_curator}"
PGDATA="${PROJECT_ROOT}/pgdata"

# Container images
BACKEND_SIF="${SIF_DIR}/backend.sif"
FRONTEND_SIF="${SIF_DIR}/frontend.sif"
DB_SIF="${SIF_DIR}/pgvector.sif"

# PostgreSQL runtime directory (for socket files)
PG_RUN_DIR="${PROJECT_ROOT}/run/postgresql"

# Ensure directories exist
mkdir -p "$SIF_DIR" "$LOG_DIR" "$RUN_DIR" "$PGDATA" "$PG_RUN_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

check_sif() {
    local sif="$1"
    local name="$2"
    if [[ ! -f "$sif" ]]; then
        log "ERROR: $name container not found at $sif"
        log "Run 'make singularity-build' first"
        return 1
    fi
}

wait_for_port() {
    local port="$1"
    local name="$2"
    local max_wait="${3:-30}"
    local waited=0
    
    log "Waiting for $name on port $port..."
    while ! nc -z localhost "$port" 2>/dev/null; do
        sleep 1
        waited=$((waited + 1))
        if [[ $waited -ge $max_wait ]]; then
            log "ERROR: $name did not start within ${max_wait}s"
            return 1
        fi
    done
    log "$name is ready on port $port"
}

start_db() {
    if singularity instance list | grep -q "paper-curator-db"; then
        log "Database already running"
        return 0
    fi
    
    check_sif "$DB_SIF" "Database" || return 1
    
    log "Starting PostgreSQL database..."
    
    # Initialize database if needed
    if [[ ! -f "$PGDATA/PG_VERSION" ]]; then
        log "Initializing database..."
        singularity exec \
            --bind "$PGDATA:/var/lib/postgresql/data" \
            --bind "$PG_RUN_DIR:/var/run/postgresql" \
            --env "PGDATA=/var/lib/postgresql/data" \
            "$DB_SIF" /usr/lib/postgresql/16/bin/initdb -D /var/lib/postgresql/data
        
        # Configure PostgreSQL to listen on all interfaces
        echo "listen_addresses = '*'" >> "$PGDATA/postgresql.conf"
        echo "port = $DB_PORT" >> "$PGDATA/postgresql.conf"
        # Use Unix socket in the runtime directory
        echo "unix_socket_directories = '/var/run/postgresql'" >> "$PGDATA/postgresql.conf"
        
        # Allow connections from localhost
        echo "host all all 127.0.0.1/32 trust" >> "$PGDATA/pg_hba.conf"
        echo "host all all ::1/128 trust" >> "$PGDATA/pg_hba.conf"
        
        # Copy init script for later execution
        cp "${PROJECT_ROOT}/src/backend/init.sql" "$PGDATA/"
    fi
    
    # Start the instance with writable directories
    singularity instance start \
        --bind "$PGDATA:/var/lib/postgresql/data" \
        --bind "$PG_RUN_DIR:/var/run/postgresql" \
        --env "PGDATA=/var/lib/postgresql/data" \
        --env "POSTGRES_USER=$DB_USER" \
        --env "POSTGRES_PASSWORD=$DB_PASS" \
        --env "POSTGRES_DB=$DB_NAME" \
        "$DB_SIF" paper-curator-db
    
    # Start PostgreSQL server inside the instance
    log "Starting PostgreSQL server..."
    singularity exec instance://paper-curator-db \
        /usr/lib/postgresql/16/bin/pg_ctl -D /var/lib/postgresql/data -l /var/lib/postgresql/data/logfile start
    
    # Wait for database to be ready
    wait_for_port "$DB_PORT" "Database" 60
    
    # Create database user and run init.sql on first run
    if [[ -f "$PGDATA/init.sql" ]]; then
        log "Creating database user and database..."
        
        # Get the OS username (this is the superuser created by initdb)
        local os_user
        os_user=$(whoami)
        
        # Create the curator user and database
        singularity exec instance://paper-curator-db \
            /usr/lib/postgresql/16/bin/psql -h localhost -p "$DB_PORT" -U "$os_user" -d postgres \
            -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASS' SUPERUSER;" 2>/dev/null || true
        
        singularity exec instance://paper-curator-db \
            /usr/lib/postgresql/16/bin/psql -h localhost -p "$DB_PORT" -U "$os_user" -d postgres \
            -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;" 2>/dev/null || true
        
        log "Running database schema initialization..."
        # Run init.sql using the curator user on the paper_curator database
        cat "${PROJECT_ROOT}/src/backend/init.sql" | singularity exec instance://paper-curator-db \
            /usr/lib/postgresql/16/bin/psql -h localhost -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME"
        
        # Remove init.sql marker to avoid re-running
        rm -f "$PGDATA/init.sql"
    fi
}

start_backend() {
    if [[ -f "${RUN_DIR}/backend.pid" ]] && kill -0 "$(cat "${RUN_DIR}/backend.pid")" 2>/dev/null; then
        log "Backend already running"
        return 0
    fi
    
    check_sif "$BACKEND_SIF" "Backend" || return 1
    
    # Ensure storage directory exists
    mkdir -p "${PROJECT_ROOT}/storage"
    
    log "Starting backend on port $BACKEND_PORT..."
    
    # Start the backend as a background process (not as instance)
    # because Singularity instances don't reliably run startscripts
    # Bind src/backend -> /app first, then overlay config + storage
    singularity run \
        --bind "${PROJECT_ROOT}/src/backend:/app" \
        --bind "${PROJECT_ROOT}/config:/app/config" \
        --bind "${PROJECT_ROOT}/storage:/app/storage" \
        --bind "${HOME}:/host_home:ro" \
        --env "BACKEND_PORT=$BACKEND_PORT" \
        --env "PYTHONUNBUFFERED=1" \
        "$BACKEND_SIF" > "${LOG_DIR}/backend.log" 2>&1 &
    
    echo $! > "${RUN_DIR}/backend.pid"
    
    wait_for_port "$BACKEND_PORT" "Backend"
}

start_frontend() {
    if [[ -f "${RUN_DIR}/frontend.pid" ]] && kill -0 "$(cat "${RUN_DIR}/frontend.pid")" 2>/dev/null; then
        log "Frontend already running"
        return 0
    fi
    
    check_sif "$FRONTEND_SIF" "Frontend" || return 1
    
    log "Starting frontend on port $FRONTEND_PORT..."
    
    # Start the frontend as a background process
    singularity run \
        --env "PORT=$FRONTEND_PORT" \
        --env "HOSTNAME=0.0.0.0" \
        "$FRONTEND_SIF" > "${LOG_DIR}/frontend.log" 2>&1 &
    
    echo $! > "${RUN_DIR}/frontend.pid"
    
    wait_for_port "$FRONTEND_PORT" "Frontend"
}

stop_service() {
    local name="$1"
    if singularity instance list | grep -q "$name"; then
        log "Stopping $name..."
        singularity instance stop "$name"
    else
        log "$name is not running"
    fi
}

stop_pid_service() {
    local name="$1"
    local pid_file="${RUN_DIR}/${name}.pid"
    
    if [[ -f "$pid_file" ]]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            log "Stopping $name (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
            sleep 2
            # Force kill if still running
            kill -9 "$pid" 2>/dev/null || true
        else
            log "$name is not running"
        fi
        rm -f "$pid_file"
    else
        log "$name is not running"
    fi
}

stop_db() {
    if singularity instance list | grep -q "paper-curator-db"; then
        log "Stopping PostgreSQL server..."
        # Gracefully stop PostgreSQL before stopping the instance
        singularity exec instance://paper-curator-db \
            /usr/lib/postgresql/16/bin/pg_ctl -D /var/lib/postgresql/data stop -m fast 2>/dev/null || true
        sleep 2
        stop_service "paper-curator-db"
    else
        log "paper-curator-db is not running"
    fi
}

start_all() {
    log "Starting all Paper Curator services..."
    start_db
    start_backend
    start_frontend
    log "All services started successfully!"
    show_status
}

stop_all() {
    log "Stopping all Paper Curator services..."
    stop_pid_service "frontend"
    stop_pid_service "backend"
    stop_db
    log "All services stopped"
}

show_status() {
    echo ""
    echo "=== Paper Curator Service Status ==="
    echo ""
    singularity instance list | grep -E "(INSTANCE|paper-curator)" || echo "No instances running"
    echo ""
    echo "Ports:"
    echo "  Database: $DB_PORT"
    echo "  Backend:  $BACKEND_PORT"
    echo "  Frontend: $FRONTEND_PORT"
    echo ""
    
    # Check if ports are listening
    for port in $DB_PORT $BACKEND_PORT $FRONTEND_PORT; do
        if nc -z localhost "$port" 2>/dev/null; then
            echo "  Port $port: LISTENING"
        else
            echo "  Port $port: NOT LISTENING"
        fi
    done
}

show_logs() {
    local service="$1"
    case "$service" in
        db|database)
            singularity exec instance://paper-curator-db tail -f /var/lib/postgresql/data/log/*.log 2>/dev/null || \
                log "No database logs available"
            ;;
        backend)
            # Singularity instances log to stdout/stderr
            log "Backend logs not available in instance mode. Use 'singularity run' for foreground logs."
            ;;
        frontend)
            log "Frontend logs not available in instance mode. Use 'singularity run' for foreground logs."
            ;;
        *)
            log "Usage: $0 logs [db|backend|frontend]"
            ;;
    esac
}

case "${1:-}" in
    start)
        case "${2:-all}" in
            all) start_all ;;
            db|database) start_db ;;
            backend) start_backend ;;
            frontend) start_frontend ;;
            *) echo "Unknown service: $2" ;;
        esac
        ;;
    stop)
        case "${2:-all}" in
            all) stop_all ;;
            db|database) stop_service "paper-curator-db" ;;
            backend) stop_service "paper-curator-backend" ;;
            frontend) stop_service "paper-curator-frontend" ;;
            *) echo "Unknown service: $2" ;;
        esac
        ;;
    restart)
        stop_all
        sleep 2
        start_all
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "${2:-}"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs} [service]"
        echo ""
        echo "Commands:"
        echo "  start [service]   Start services (all, db, backend, frontend)"
        echo "  stop [service]    Stop services"
        echo "  restart           Restart all services"
        echo "  status            Show service status"
        echo "  logs [service]    Show logs for a service"
        exit 1
        ;;
esac
