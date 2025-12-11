-- Open Notebook Database Schema
-- Final version - no migrations, direct schema initialization
-- Excludes: embedding fields (moved to Qdrant), podcast functionality (removed)

-- =============================================================================
-- CORE TABLES
-- =============================================================================

-- Notebook table
DEFINE TABLE notebook SCHEMAFULL;
DEFINE FIELD name ON TABLE notebook TYPE option<string>;
DEFINE FIELD description ON TABLE notebook TYPE option<string>;
DEFINE FIELD created ON notebook DEFAULT time::now() VALUE $before OR time::now();
DEFINE FIELD updated ON notebook DEFAULT time::now() VALUE time::now();
DEFINE FIELD archived ON TABLE notebook TYPE option<bool> DEFAULT False;

-- Source table
DEFINE TABLE source SCHEMAFULL;
DEFINE FIELD title ON TABLE source TYPE option<string>;
DEFINE FIELD asset ON TABLE source FLEXIBLE TYPE option<object>;
DEFINE FIELD created ON source DEFAULT time::now() VALUE $before OR time::now();
DEFINE FIELD updated ON source DEFAULT time::now() VALUE time::now();
DEFINE FIELD topics ON TABLE source TYPE option<array<string>>;
DEFINE FIELD full_text ON TABLE source TYPE option<string>;
DEFINE FIELD command ON TABLE source TYPE option<record<command>>;
-- PageIndex fields for persistent storage of tree structure
DEFINE FIELD pageindex_structure ON TABLE source FLEXIBLE TYPE option<object>;
DEFINE FIELD pageindex_built_at ON TABLE source TYPE option<datetime>;
DEFINE FIELD pageindex_model ON TABLE source TYPE option<string>;
DEFINE FIELD pageindex_version ON TABLE source TYPE option<string>;

-- Transformation table
DEFINE TABLE transformation SCHEMAFULL;
DEFINE FIELD name ON TABLE transformation TYPE string;
DEFINE FIELD title ON TABLE transformation TYPE string;
DEFINE FIELD description ON TABLE transformation TYPE option<string>;
DEFINE FIELD prompt ON TABLE transformation TYPE string;
DEFINE FIELD type ON TABLE transformation TYPE string;
DEFINE FIELD model ON TABLE transformation TYPE option<string>;
DEFINE FIELD output_schema ON TABLE transformation TYPE option<object>;
DEFINE FIELD apply_default ON TABLE transformation TYPE bool DEFAULT False;
DEFINE FIELD created ON transformation DEFAULT time::now() VALUE $before OR time::now();
DEFINE FIELD updated ON transformation DEFAULT time::now() VALUE time::now();

-- Source chunk table (stores chunked content for vector search)
DEFINE TABLE source_chunk SCHEMAFULL;
DEFINE FIELD source_id ON TABLE source_chunk TYPE record<source>;
DEFINE FIELD chunk_index ON TABLE source_chunk TYPE int;
DEFINE FIELD content ON TABLE source_chunk TYPE string;
DEFINE FIELD created ON source_chunk DEFAULT time::now() VALUE $before OR time::now();
DEFINE FIELD updated ON source_chunk DEFAULT time::now() VALUE time::now();

-- Source insight table (SCHEMALESS - no embedding field)
DEFINE TABLE source_insight SCHEMALESS;

-- Chat session table (SCHEMALESS)
DEFINE TABLE chat_session SCHEMALESS;

-- =============================================================================
-- RELATIONSHIP TABLES
-- =============================================================================

-- Reference relation (many-to-many between notebooks and sources)
DEFINE TABLE reference TYPE RELATION FROM source TO notebook;

-- Refers to relation (chat sessions can refer to notebooks or sources)
DEFINE TABLE refers_to TYPE RELATION FROM chat_session TO notebook|source;

-- =============================================================================
-- METADATA TABLES
-- =============================================================================

-- Idempotency record table for API request deduplication
DEFINE TABLE idempotency_record SCHEMAFULL;

-- Core fields
DEFINE FIELD idempotency_key ON TABLE idempotency_record TYPE string;
DEFINE FIELD request_fingerprint ON TABLE idempotency_record TYPE string;
DEFINE FIELD endpoint ON TABLE idempotency_record TYPE string;
DEFINE FIELD http_method ON TABLE idempotency_record TYPE string;

-- Request/Response data
DEFINE FIELD request_body ON TABLE idempotency_record TYPE option<object>;
DEFINE FIELD response_status ON TABLE idempotency_record TYPE int;
DEFINE FIELD response_body ON TABLE idempotency_record TYPE option<object>;

-- Associated command (if applicable)
DEFINE FIELD command_id ON TABLE idempotency_record TYPE option<record<command>>;

-- Metadata
DEFINE FIELD created ON TABLE idempotency_record DEFAULT time::now() VALUE $before OR time::now();
DEFINE FIELD expires_at ON TABLE idempotency_record TYPE datetime;
DEFINE FIELD locked_until ON TABLE idempotency_record TYPE option<datetime>;

-- Processing state
DEFINE FIELD status ON TABLE idempotency_record TYPE string 
    DEFAULT "processing" 
    ASSERT $value INSIDE ["processing", "completed", "failed"];

-- =============================================================================
-- ANALYZER
-- =============================================================================

-- Define analyzer for full-text search
DEFINE ANALYZER my_analyzer TOKENIZERS blank,class,camel,punct FILTERS snowball(english), lowercase;

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Full-text search indexes
DEFINE INDEX idx_source_title ON TABLE source COLUMNS title SEARCH ANALYZER my_analyzer BM25 HIGHLIGHTS;
DEFINE INDEX idx_source_full_text ON TABLE source COLUMNS full_text SEARCH ANALYZER my_analyzer BM25 HIGHLIGHTS;
DEFINE INDEX idx_source_insight ON TABLE source_insight COLUMNS content SEARCH ANALYZER my_analyzer BM25 HIGHLIGHTS;

-- Idempotency indexes
DEFINE INDEX idx_idempotency_key ON TABLE idempotency_record COLUMNS idempotency_key UNIQUE;
DEFINE INDEX idx_expires_at ON TABLE idempotency_record COLUMNS expires_at;

-- PageIndex indexes (for querying sources with PageIndex)
DEFINE INDEX idx_pageindex_built_at ON TABLE source COLUMNS pageindex_built_at;

-- Source chunk indexes (for efficient deletion and querying)
DEFINE INDEX idx_source_id ON TABLE source_chunk COLUMNS source_id;

-- =============================================================================
-- EVENTS
-- =============================================================================

-- Source delete event (cleanup related data)
DEFINE EVENT source_delete ON TABLE source WHEN ($after == NONE) THEN {
    DELETE source_insight WHERE source == $before.id;
    DELETE source_chunk WHERE source_id == $before.id;
    -- Note: pageindex_structure is automatically deleted with the source record
    -- Application layer should clear in-memory cache via PageIndexService.clear_cache_for_source()
};

-- =============================================================================
-- FUNCTIONS
-- =============================================================================

-- Text search function (no embedding references)
DEFINE FUNCTION fn::text_search($query_text: string, $match_count: int, $sources: bool, $show_notes: bool) {
    -- Note: $show_notes parameter kept for backward compatibility but ignored (always false)
    let $source_title_search = 
        IF $sources {(
            SELECT id as item_id, math::max(search::score(1)) AS relevance
            FROM source
            WHERE title @1@ $query_text
            GROUP BY item_id)}
        ELSE { [] };
    
    let $source_full_search = 
         IF $sources {(
            SELECT source as item_id, math::max(search::score(1)) AS relevance
            FROM source
            WHERE full_text @1@ $query_text
            GROUP BY item_id)}
        ELSE { [] };
    
    let $source_insight_search = 
         IF $sources {(
             SELECT source as item_id, math::max(search::score(1)) AS relevance
            FROM source_insight
            WHERE content @1@ $query_text
            GROUP BY item_id)}
        ELSE { [] };

    let $source_asset_results = array::union($source_title_search, $source_insight_search);
    let $source_results = array::union($source_full_search, $source_asset_results);
    let $final_results = $source_results;

    RETURN (SELECT item_id, math::max(relevance) as relevance from $final_results
        GROUP BY item_id ORDER BY relevance DESC LIMIT $match_count);
};
