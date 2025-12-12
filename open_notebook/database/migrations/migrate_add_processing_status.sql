-- Migration: Add processing_status and error_message fields to source table
-- Date: 2024
-- Description: Adds fields to track source processing status and error messages
--              for better user experience and error handling

-- Note: SurrealDB doesn't support ALTER TABLE, so these fields should be added
-- via schema.sql. This migration script is for updating existing records.

-- Set default processing_status to 'completed' for existing sources
UPDATE source SET processing_status = 'completed' WHERE processing_status IS NONE;

-- Clear any existing error messages (optional - only if you want to clean up)
-- UPDATE source SET error_message = NONE WHERE error_message IS NOT NONE;

