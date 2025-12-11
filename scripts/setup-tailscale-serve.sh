#!/bin/bash
# Tailscale Serve Setup Script (Linux/Mac Host)

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================"
echo "=== Tailscale Serve Setup Script ==="
echo "========================================"
echo ""

# Check if Tailscale is installed
if ! command -v tailscale &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Tailscale is not installed"
    echo "Please install Tailscale first: https://tailscale.com/download"
    exit 1
fi

# Check if Tailscale is connected
if ! tailscale status &> /dev/null; then
    echo -e "${YELLOW}[WARNING]${NC} Tailscale is not connected"
    echo "Please connect Tailscale first"
    exit 1
fi

# Read configuration from docker.env (if exists)
FRONTEND_PORT=8188
API_PORT=5055
EXPOSE_API=false

if [ -f "docker.env" ]; then
    echo "Reading docker.env configuration..."
    # Read environment variables from docker.env, ignoring comments and empty lines
    while IFS='=' read -r key value || [ -n "$key" ]; do
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        
        # Remove quotes if present
        value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
        
        case "$key" in
            TAILSCALE_FRONTEND_PORT)
                FRONTEND_PORT="$value"
                ;;
            TAILSCALE_API_PORT)
                API_PORT="$value"
                ;;
            TAILSCALE_EXPOSE_API)
                EXPOSE_API="$value"
                ;;
        esac
    done < docker.env
fi

echo "Frontend Port: $FRONTEND_PORT"
echo "API Port: $API_PORT"
echo ""

# Get Tailscale information
echo "Getting Tailscale information..."

# Get Tailscale IP
TAILSCALE_IP=$(tailscale ip -4 2>/dev/null | head -n1 || echo "")

echo -e "${GREEN}[OK]${NC} Tailscale is connected"
if [ -n "$TAILSCALE_IP" ]; then
    echo "  Tailscale IP: $TAILSCALE_IP"
fi
echo ""

# Check if services are running
echo "Checking service status..."

# Function to check if port is open
check_port() {
    local port=$1
    local service_name=$2
    
    # Use netcat (nc) or ss to check port
    if command -v nc &> /dev/null; then
        if nc -z localhost "$port" 2>/dev/null; then
            echo -e "${GREEN}[OK]${NC} $service_name is running (port $port)"
            return 0
        else
            echo -e "${YELLOW}[WARNING]${NC} $service_name is not running (port $port)"
            return 1
        fi
    elif command -v ss &> /dev/null; then
        if ss -lnt | grep -q ":$port "; then
            echo -e "${GREEN}[OK]${NC} $service_name is running (port $port)"
            return 0
        else
            echo -e "${YELLOW}[WARNING]${NC} $service_name is not running (port $port)"
            return 1
        fi
    else
        # Fallback: try to connect using /dev/tcp
        if timeout 1 bash -c "echo >/dev/tcp/localhost/$port" 2>/dev/null; then
            echo -e "${GREEN}[OK]${NC} $service_name is running (port $port)"
            return 0
        else
            echo -e "${YELLOW}[WARNING]${NC} $service_name is not running (port $port)"
            return 1
        fi
    fi
}

# Check frontend port
SERVICE_READY=0
if ! check_port "$FRONTEND_PORT" "Frontend service"; then
    SERVICE_READY=1
fi

# Wait for service to start (if needed)
if [ $SERVICE_READY -ne 0 ]; then
    MAX_WAIT=60
    WAITED=0
    echo "Waiting for frontend service to start (max $MAX_WAIT seconds)..."
    
    while [ $WAITED -lt $MAX_WAIT ]; do
        # Check port using available method
        if command -v nc &> /dev/null; then
            if nc -z localhost "$FRONTEND_PORT" 2>/dev/null; then
                echo ""
                echo -e "${GREEN}[OK]${NC} Frontend service is ready"
                SERVICE_READY=0
                break
            fi
        elif command -v ss &> /dev/null; then
            if ss -lnt | grep -q ":$FRONTEND_PORT "; then
                echo ""
                echo -e "${GREEN}[OK]${NC} Frontend service is ready"
                SERVICE_READY=0
                break
            fi
        else
            if timeout 1 bash -c "echo >/dev/tcp/localhost/$FRONTEND_PORT" 2>/dev/null; then
                echo ""
                echo -e "${GREEN}[OK]${NC} Frontend service is ready"
                SERVICE_READY=0
                break
            fi
        fi
        
        sleep 2
        WAITED=$((WAITED + 2))
        echo -n "."
    done
    
    if [ $SERVICE_READY -ne 0 ]; then
        echo ""
        echo -e "${RED}[ERROR]${NC} Frontend service startup timeout"
        echo "Please ensure Docker containers are started and services are running"
        exit 1
    fi
fi

echo ""

# Check API port (if enabled)
if [ "$EXPOSE_API" = "true" ]; then
    check_port "$API_PORT" "API service"
    echo ""
fi

# Show current Tailscale Serve status
echo "Current Tailscale Serve configuration:"
tailscale serve status 2>/dev/null || echo "  Not configured yet"
echo ""

# Configure Tailscale Serve
echo "Configuring Tailscale Serve..."
echo ""

# Check if sudo/root privileges are needed
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}[WARNING]${NC} Root privileges required to configure Tailscale Serve"
    echo "Please run this script with sudo or as root"
    echo ""
    echo "Or manually execute the following command (HTTPS only, recommended):"
    echo "  sudo tailscale serve --bg --https 443 $FRONTEND_PORT"
    echo ""
    echo "Or if you need both HTTP and HTTPS:"
    echo "  sudo tailscale serve --bg --http 80 $FRONTEND_PORT"
    echo "  sudo tailscale serve --bg --https 443 $FRONTEND_PORT"
    echo ""
    read -p "Press Enter to continue..."
    exit 1
fi

# Configure frontend Serve
echo "Configuring frontend service (https:// -> http://localhost:$FRONTEND_PORT)..."
echo "Mapping root path to local service..."
# Reset any existing configuration first to ensure clean setup
tailscale serve reset >/dev/null 2>&1 || true
# Use new Tailscale Serve syntax (v1.90+)
# Direct port number syntax: Tailscale automatically maps to root path (/)
# --https 443: maps to HTTPS port 443 (standard HTTPS, recommended for security)
# --bg: run in background
# Using port number directly (8188) instead of full URL to avoid path mapping issues
# Note: Only HTTPS is configured by default for security. HTTP (port 80) is optional.
tailscale serve --bg --https 443 $FRONTEND_PORT

if [ $? -eq 0 ]; then
    echo -e "${GREEN}[OK]${NC} Serve configuration successful"
else
    echo -e "${RED}[ERROR]${NC} Serve configuration failed, please verify:"
    echo "  1. Serve feature is enabled in Tailscale admin console"
    echo "  2. You have sufficient permissions to execute tailscale commands"
    exit 1
fi

echo ""
echo "========================================"
echo -e "${GREEN}[OK]${NC} Tailscale Serve configuration completed!"
echo "========================================"
echo ""

# Display access information
echo "=== Access Information ==="
if [ -n "$TAILSCALE_IP" ]; then
    echo "Using Tailscale IP:"
    echo "  https://$TAILSCALE_IP"
    echo ""
fi

# Try to get MagicDNS domain
TAILSCALE_DOMAIN=$(tailscale status --self 2>/dev/null | grep -oE '[a-zA-Z0-9-]+\.tail[\w-]+\.ts\.net' | head -n1 || echo "")

if [ -n "$TAILSCALE_DOMAIN" ]; then
    echo "MagicDNS address:"
    echo "  https://$TAILSCALE_DOMAIN"
    echo ""
fi

echo "=== Current Serve Status ==="
tailscale serve status

echo ""
echo "Tips:"
echo "  - To stop: tailscale serve reset"
echo "  - To check status: tailscale serve status"
echo "  - To view details: tailscale status"
echo ""
