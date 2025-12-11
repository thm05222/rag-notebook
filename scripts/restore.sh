#!/bin/bash
# Restore script for RAG Notebook data
# Restores SurrealDB, Qdrant, or notebook data from backups

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_DIR="${PROJECT_ROOT}/backups"

# Function to list available backups
list_backups() {
    local backup_type="$1"
    echo -e "${GREEN}Available ${backup_type} backups:${NC}"
    ls -lh "$BACKUP_DIR"/*${backup_type}*.tar.gz 2>/dev/null | awk '{print $9, "(" $5 ")"}' || {
        echo -e "${YELLOW}No ${backup_type} backups found${NC}"
        return 1
    }
}

# Function to restore a backup
restore_backup() {
    local backup_file="$1"
    local target_dir="$2"
    local backup_name="$3"
    
    if [ ! -f "$backup_file" ]; then
        echo -e "${RED}Error: Backup file not found: $backup_file${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}WARNING: This will replace existing data in $target_dir${NC}"
    echo "Press Ctrl+C to cancel, or Enter to continue..."
    read -r
    
    # Stop services if running
    if docker compose ps 2>/dev/null | grep -q "rag-notebook"; then
        echo -e "${YELLOW}Stopping services...${NC}"
        docker compose down || true
    fi
    
    # Backup current data first (safety measure)
    if [ -d "$target_dir" ] && [ "$(ls -A "$target_dir" 2>/dev/null)" ]; then
        SAFE_BACKUP="${target_dir}_pre_restore_$(date +%Y%m%d_%H%M%S)"
        echo "Creating safety backup: $SAFE_BACKUP"
        mv "$target_dir" "$SAFE_BACKUP" || {
            echo -e "${RED}Failed to create safety backup${NC}"
            return 1
        }
    fi
    
    # Extract backup
    echo "Restoring $backup_name from $backup_file..."
    mkdir -p "$target_dir"
    tar -xzf "$backup_file" -C "$PROJECT_ROOT" || {
        echo -e "${RED}Failed to extract backup${NC}"
        return 1
    }
    
    echo -e "${GREEN}✓ Restored $backup_name successfully${NC}"
    echo "You may need to restart services: docker compose up -d"
}

# Main menu
if [ $# -eq 0 ]; then
    echo "RAG Notebook Restore Script"
    echo ""
    echo "Usage:"
    echo "  $0 list [surreal|qdrant|notebook]  - List available backups"
    echo "  $0 restore <backup_file>            - Restore from backup file"
    echo ""
    echo "Examples:"
    echo "  $0 list surreal                     - List SurrealDB backups"
    echo "  $0 restore backups/surreal_backup_20250102_120000.tar.gz"
    exit 0
fi

case "$1" in
    list)
        backup_type="${2:-}"
        if [ -z "$backup_type" ]; then
            list_backups "surreal"
            echo ""
            list_backups "qdrant"
            echo ""
            list_backups "notebook"
        else
            list_backups "$backup_type"
        fi
        ;;
    restore)
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Please specify backup file${NC}"
            echo "Usage: $0 restore <backup_file>"
            exit 1
        fi
        
        backup_file="$2"
        if [[ "$backup_file" == *"surreal"* ]]; then
            restore_backup "$backup_file" "${PROJECT_ROOT}/surreal_data" "SurrealDB"
        elif [[ "$backup_file" == *"qdrant"* ]]; then
            restore_backup "$backup_file" "${PROJECT_ROOT}/qdrant_data" "Qdrant"
        elif [[ "$backup_file" == *"notebook"* ]]; then
            restore_backup "$backup_file" "${PROJECT_ROOT}/notebook_data" "Notebook data"
        else
            echo -e "${RED}Error: Cannot determine backup type from filename${NC}"
            exit 1
        fi
        ;;
    *)
        echo -e "${RED}Error: Unknown command: $1${NC}"
        exit 1
        ;;
esac

