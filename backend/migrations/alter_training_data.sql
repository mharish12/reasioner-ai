-- Migration: Update training_data table column sizes
-- This fixes the StringDataRightTruncation error for long filenames

-- Increase document_type column size from VARCHAR(20) to VARCHAR(50)
ALTER TABLE training_data ALTER COLUMN document_type TYPE VARCHAR(50);

-- Increase document_name column size from VARCHAR(200) to VARCHAR(500)
ALTER TABLE training_data ALTER COLUMN document_name TYPE VARCHAR(500);

-- Verify the changes
SELECT 
    column_name, 
    data_type, 
    character_maximum_length 
FROM information_schema.columns 
WHERE table_name = 'training_data' 
    AND column_name IN ('document_type', 'document_name');

