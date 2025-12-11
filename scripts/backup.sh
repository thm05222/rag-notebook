#!/bin/bash
# Backup script for RAG Notebook data
# Backs up SurrealDB, Qdrant, and notebook data

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_DIR="${PROJECT_ROOT}/backups"
DATE=$(date +%Y%m%d_%H%M%S)
MAX_BACKUPS_DAYS=${MAX_BACKUPS_DAYS:-7}  # Keep backups for 7 days by default

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting backup process...${NC}"
echo "Backup directory: $BACKUP_DIR"
echo "Date: $DATE"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Function to backup a directory
backup_directory() {
    local source_dir="$1"
    local backup_name="$2"
    local backup_file="${BACKUP_DIR}/${backup_name}_${DATE}.tar.gz"
    
    if [ ! -d "$source_dir" ]; then
        echo -e "${YELLOW}Warning: $source_dir does not exist, skipping...${NC}"
        return 0
    fi
    
    if [ ! "$(ls -A "$source_dir" 2>/dev/null)" ]; then
        echo -e "${YELLOW}Warning: $source_dir is empty, skipping...${NC}"
        return 0
    fi
    
    echo "Backing up $source_dir to $backup_file..."
    tar -czf "$backup_file" -C "$PROJECT_ROOT" "$(basename "$source_dir")" 2>/dev/null || {
        echo -e "${RED}Error: Failed to backup $source_dir${NC}"
        return 1
    }
    
    local size=$(du -h "$backup_file" | cut -f1)
    echo -e "${GREEN}✓ Backed up $backup_name (size: $size)${NC}"
}

# Function to check disk space
check_disk_space() {
    local available=$(df -BG "$BACKUP_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ "$available" -lt 1 ]; then
        echo -e "${RED}Error: Less than 1GB available disk space${NC}"
        exit 1
    fi
}

# Check disk space before backup
check_disk_space

# Backup SurrealDB data
backup_directory "${PROJECT_ROOT}/surreal_data" "surreal_backup"

# Backup Qdrant data
backup_directory "${PROJECT_ROOT}/qdrant_data" "qdrant_backup"

# Backup notebook data
backup_directory "${PROJECT_ROOT}/notebook_data" "notebook_backup"

# Clean up old backups
echo -e "${GREEN}Cleaning up backups older than $MAX_BACKUPS_DAYS days...${NC}"
find "$BACKUP_DIR" -name "*.tar.gz" -type f -mtime +$MAX_BACKUPS_DAYS -delete || true
OLD_COUNT=$(find "$BACKUP_DIR" -name "*.tar.gz" -type f -mtime +$MAX_BACKUPS_DAYS | wc -l)
if [ "$OLD_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}Removed $OLD_COUNT old backup(s)${NC}"
else
    echo "No old backups to remove"
fi

# Summary
BACKUP_COUNT=$(find "$BACKUP_DIR" -name "*.tar.gz" -type f | wc -l)
TOTAL_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)

echo ""
echo -e "${GREEN}Backup completed successfully!${NC}"
echo "Total backups: $BACKUP_COUNT"
echo "Total backup size: $TOTAL_SIZE"
echo "Backup location: $BACKUP_DIR"

