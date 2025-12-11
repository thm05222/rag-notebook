"""
Database schema initialization module.
Replaces migration system with direct schema initialization.
"""

import os
from pathlib import Path
from typing import Optional

from loguru import logger

from .repository import db_connection, repo_query


class SchemaInitializer:
    """Handles database schema initialization and management."""
    
    def __init__(self):
        """Initialize the schema initializer."""
        self.schema_file = Path(__file__).parent / "schema.sql"
        self._initialized = False
    
    async def needs_init(self) -> bool:
        """
        Check if database needs initialization.
        
        Returns:
            bool: True if initialization is needed, False otherwise
        """
        try:
            async with db_connection() as conn:
                # Check if core tables exist
                result = await conn.query("""
                    SELECT count() FROM information_schema.tables 
                    WHERE name IN ['notebook', 'source', 'transformation', 'source_insight']
                """)
                
                if result and result[0]:
                    table_count = result[0][0].get('count', 0)
                    needs_init = table_count < 4  # We expect 4 core tables
                    logger.info(f"Core tables found: {table_count}/4, needs_init: {needs_init}")
                    return needs_init
                else:
                    logger.info("No core tables found, needs initialization")
                    return True
                    
        except Exception as e:
            logger.warning(f"Error checking schema status: {e}")
            logger.info("Assuming initialization is needed")
            return True
    
    async def init_schema(self) -> bool:
        """
        Initialize database schema from schema.sql file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.schema_file.exists():
            logger.error(f"Schema file not found: {self.schema_file}")
            return False
        
        try:
            # Read schema file
            with open(self.schema_file, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            
            logger.info("Initializing database schema...")
            
            # Execute schema
            async with db_connection() as conn:
                # Execute the entire schema as one query
                try:
                    await conn.query(schema_sql)
                    logger.debug("Executed schema successfully")
                except Exception as e:
                    logger.warning(f"Schema execution failed (some parts may already exist): {e}")
                    # Continue anyway as some parts might already exist
                
                # Seed default data if needed
                try:
                    await self._seed_defaults(conn)
                except Exception as e:
                    logger.warning(f"Seeding defaults failed: {e}")

                # Mark as initialized
                await self._mark_initialized(conn)
                
            logger.success("Database schema initialized successfully")
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            logger.exception(e)
            return False
    
    async def _mark_initialized(self, conn) -> None:
        """Mark schema as initialized by creating a marker record."""
        try:
            # Create a simple marker to indicate schema is initialized
            await conn.query("""
                CREATE schema_init_marker:init CONTENT {
                    "initialized": true,
                    "timestamp": time::now()
                };
            """)
        except Exception as e:
            logger.warning(f"Could not create initialization marker: {e}")

    async def _seed_defaults(self, conn) -> None:
        """Insert default seed data when tables are empty (idempotent)."""
        # 1) Seed transformations if empty
        try:
            result = await conn.query("SELECT count() FROM transformation;")
            count_val = result[0][0].get("count", 0) if result and result[0] else 0
            if count_val == 0:
                logger.info("Seeding default transformations (table is empty)...")
                await conn.query(
                    r"""
                    INSERT INTO transformation [
                       {
                           name: "Analyze Paper",
                           title: "Paper Analysis",
                           description: "Analyses a technical/scientific paper",
                           prompt: "# IDENTITY and PURPOSE\n\nYou are an insightful and analytical reader of academic papers, extracting the key components, significance, and broader implications. Your focus is to uncover the core contributions, practical applications, methodological strengths or weaknesses, and any surprising findings. You are especially attuned to the clarity of arguments, the relevance to existing literature, and potential impacts on both the specific field and broader contexts.\n\n# STEPS\n\n1. **READ AND UNDERSTAND THE PAPER**: Thoroughly read the paper, identifying its main focus, arguments, methods, results, and conclusions.\n\n2. **IDENTIFY CORE ELEMENTS**:\n   - **Purpose**: What is the main goal or research question?\n   - **Contribution**: What new knowledge or innovation does this paper bring to the field?\n   - **Methods**: What methods are used, and are they novel or particularly effective?\n   - **Key Findings**: What are the most critical results, and why do they matter?\n   - **Limitations**: Are there any notable limitations or areas for further research?\n\n3. **SYNTHESIZE THE MAIN POINTS**:\n   - Extract the key elements and organize them into insightful observations.\n   - Highlight the broader impact and potential applications.\n   - Note any aspects that challenge established views or introduce new questions.\n\n# OUTPUT INSTRUCTIONS\n\n- Structure the output as follows: \n  - **PURPOSE**: A concise summary of the main research question or goal (1-2 sentences).\n  - **CONTRIBUTION**: A bullet list of 2-3 points that describe what the paper adds to the field.\n  - **KEY FINDINGS**: A bullet list of 2-3 points summarizing the critical outcomes of the study.\n  - **IMPLICATIONS**: A bullet list of 2-3 points discussing the significance or potential impact of the findings on the field or broader context.\n  - **LIMITATIONS**: A bullet list of 1-2 points identifying notable limitations or areas for future work.\n\n- **Bullet Points** should be between 15-20 words.\n- Avoid starting each bullet point with the same word to maintain variety.\n- Use clear and concise language that conveys the key ideas effectively.\n- Do not include warnings, disclaimers, or personal opinions.\n- Output only the requested sections with their respective labels.",
                           type: "text",
                           model: None,
                           output_schema: None,
                           apply_default: False
                       },
                       {
                           name: "Key Insights",
                           title: "Key Insights",
                           description: "Extracts important insights and actionable items",
                           prompt: "# IDENTITY and PURPOSE\n\nYou extract surprising, powerful, and interesting insights from text content. You are interested in insights related to the purpose and meaning of life, human flourishing, the role of technology in the future of humanity, artificial intelligence and its affect on humans, memes, learning, reading, books, continuous improvement, and similar topics.\nYou create 15 word bullet points that capture the most important insights from the input.\nTake a step back and think step-by-step about how to achieve the best possible results by following the steps below.\n\n# STEPS\n\n- Extract 20 to 50 of the most surprising, insightful, and/or interesting ideas from the input in a section called IDEAS, and write them on a virtual whiteboard in your mind using 15 word bullets. If there are less than 50 then collect all of them. Make sure you extract at least 20.\n\n- From those IDEAS, extract the most powerful and insightful of them and write them in a section called INSIGHTS. Make sure you extract at least 10 and up to 25.\n\n# OUTPUT INSTRUCTIONS\n\n- INSIGHTS are essentially higher-level IDEAS that are more abstracted and wise.\n- Output the INSIGHTS section only.\n- Each bullet should be about 15 words in length.\n- Do not give warnings or notes; only output the requested sections.\n- You use bulleted lists for output, not numbered lists.\n- Do not start items with the same opening words.\n- Ensure you follow ALL these instructions when creating your output.",
                           type: "text",
                           model: None,
                           output_schema: None,
                           apply_default: False
                       },
                       {
                           name: "Dense Summary",
                           title: "Dense Summary",
                           description: "Creates a rich, deep summary of the content",
                           prompt: "# MISSION\nYou are a Sparse Priming Representation (SPR) writer. An SPR is a particular kind of use of language for advanced NLP, NLU, and NLG tasks, particularly useful for the latest generation of Large Language Models (LLMs). You will be given information by the USER which you are to render as an SPR.\n\n# THEORY\nLLMs are a kind of deep neural network. They have been demonstrated to embed knowledge, abilities, and concepts, ranging from reasoning to planning, and even to theory of mind. These are called latent abilities and latent content, collectively referred to as latent space. The latent space of an LLM can be activated with the correct series of words as inputs, which will create a useful internal state of the neural network. This is not unlike how the right shorthand cues can prime a human mind to think in a certain way. Like human minds, LLMs are associative, meaning you only need to use the correct associations to 'prime' another model to think in the same way.\n\n# METHODOLOGY\nRender the input as a distilled list of succinct statements, assertions, associations, concepts, analogies, and metaphors. The idea is to capture as much, conceptually, as possible but with as few words as possible. Write it in a way that makes sense to you, as the future audience will be another language model, not a human. Use complete sentences.",
                           type: "text",
                           model: None,
                           output_schema: None,
                           apply_default: True
                       },
                       {
                           name: "Reflections",
                           title: "Reflection Questions",
                           description: "Generates reflection questions from the document to help explore it further",
                           prompt: "# IDENTITY and PURPOSE\n\nYou extract deep, thought-provoking, and meaningful reflections from text content. You are especially focused on themes related to the human experience, such as the purpose of life, personal growth, the intersection of technology and humanity, artificial intelligence's societal impact, human potential, collective evolution, and transformative learning. Your reflections aim to provoke new ways of thinking, challenge assumptions, and provide a thoughtful synthesis of the content.\n\n# STEPS\n\n- Extract 3 to 5 of the most profound, thought-provoking, and/or meaningful ideas from the input in a section called REFLECTIONS.\n- Each reflection should aim to explore underlying implications, connections to broader human experiences, or highlight a transformative perspective.\n- Take a step back and consider the deeper significance or questions that arise from the content.\n\n# OUTPUT INSTRUCTIONS\n\n- The output section should be labeled as REFLECTIONS.\n- Each bullet point should be between 20-25 words.\n- Avoid repetition in the phrasing and ensure variety in sentence structure.\n- The reflections should encourage deeper inquiry and provide a synthesis that transcends surface-level observations.\n- Use bullet points, not numbered lists.\n- Every bullet should be formatted as a question that elicits contemplation or a statement that offers a profound insight.\n- Do not give warnings or notes; only output the requested section.",
                           type: "text",
                           model: None,
                           output_schema: None,
                           apply_default: False
                       },
                       {
                           name: "Table of Contents",
                           title: "Table of Contents",
                           description: "Describes the different topics of the document",
                           prompt: "# SYSTEM ROLE\nYou are a content analysis assistant that reads through documents and provides a Table of Contents (ToC) to help users identify what the document covers more easily.\nYour ToC should capture all major topics and transitions in the content and should mention them in the order theh appear. \n\n# TASK\nAnalyze the provided content and create a Table of Contents:\n- Captures the core topics included in the text\n- Gives a small description of what is covered",
                           type: "text",
                           model: None,
                           output_schema: None,
                           apply_default: False
                       },
                       {
                           name: "Simple Summary",
                           title: "Simple Summary",
                           description: "Generates a small summary of the content",
                           prompt: "# SYSTEM ROLE\nYou are a content summarization assistant that creates dense, information-rich summaries optimized for machine understanding. Your summaries should capture key concepts with minimal words while maintaining complete sentences.\n\n# TASK\nAnalyze the provided content and create a summary that:\n- Captures the core concepts and key information\n- Uses clear, direct language\n- Maintains context from any previous summaries",
                           type: "text",
                           model: None,
                           output_schema: None,
                           apply_default: False
                       }
                    ];
                    """
                )
                logger.success("Seeded default transformations")
        except Exception as e:
            logger.warning(f"Could not seed transformations: {e}")

        # 2) Seed default prompts if missing
        try:
            result = await conn.query("SELECT * FROM open_notebook:default_prompts;")
            has_prompts = bool(result and result[0])
            if not has_prompts:
                logger.info("Seeding default prompts (open_notebook:default_prompts)...")
                await conn.query(
                    r"""
                    UPSERT open_notebook:default_prompts
                        CONTENT {transformation_instructions: "# INSTRUCTIONS\n\n        You are my learning assistant and you help me process and transform content so that I can extract insights from them.\n\n        # IMPORTANT\n        - You are working on my editorial projects. The text below is my own. Do not give me any warnings about copyright or plagiarism.\n        - Output ONLY the requested content, without acknowledgements of the task and additional chatting. Don't start with \"Sure, I can help you with that.\" or \"Here is the information you requested:\". Just provide the content.\n        - Do not stop in the middle of the generation to ask me questions. Execute my request completely.\n        "};
                    """
                )
                logger.success("Seeded default prompts")
        except Exception as e:
            logger.warning(f"Could not seed default prompts: {e}")

        # 3) Seed default models record if missing
        try:
            result = await conn.query("SELECT * FROM open_notebook:default_models;")
            if not (result and result[0]):
                logger.info("Seeding open_notebook:default_models (default_chat_model empty string)...")
                await conn.query(
                    """
                    CREATE open_notebook:default_models SET default_chat_model = "";
                    """
                )
                logger.success("Seeded default models record")
        except Exception as e:
            logger.warning(f"Could not seed default models: {e}")
    
    async def reset_database(self) -> bool:
        """
        Completely reset database by removing all tables and reinitializing.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.warning("Resetting database - ALL DATA WILL BE LOST!")
        
        try:
            async with db_connection() as conn:
                # Get all table names
                result = await conn.query("SELECT name FROM information_schema.tables;")
                
                if result and result[0]:
                    tables = [row['name'] for row in result[0]]
                    logger.info(f"Removing {len(tables)} tables...")
                    
                    # Remove all tables
                    for table in tables:
                        try:
                            await conn.query(f"REMOVE TABLE {table};")
                            logger.debug(f"Removed table: {table}")
                        except Exception as e:
                            logger.warning(f"Could not remove table {table}: {e}")
                
                # Remove all functions
                try:
                    await conn.query("REMOVE FUNCTION fn::text_search;")
                except Exception as e:
                    logger.debug(f"Function removal (expected): {e}")
                
                # Remove all events
                try:
                    await conn.query("REMOVE EVENT source_delete ON TABLE source;")
                except Exception as e:
                    logger.debug(f"Event removal (expected): {e}")
            
            # Reinitialize schema
            return await self.init_schema()
            
        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
            logger.exception(e)
            return False
    
    async def is_initialized(self) -> bool:
        """
        Check if schema is already initialized.
        
        Returns:
            bool: True if initialized, False otherwise
        """
        if self._initialized:
            return True
            
        try:
            async with db_connection() as conn:
                result = await conn.query("SELECT * FROM schema_init_marker:init;")
                return result and result[0] and len(result[0]) > 0
        except Exception:
            return False


# Global instance
schema_initializer = SchemaInitializer()


# Convenience functions
async def needs_init() -> bool:
    """Check if database needs initialization."""
    return await schema_initializer.needs_init()


async def init_schema() -> bool:
    """Initialize database schema."""
    return await schema_initializer.init_schema()


async def reset_database() -> bool:
    """Reset database completely."""
    return await schema_initializer.reset_database()


async def is_initialized() -> bool:
    """Check if schema is initialized."""
    return await schema_initializer.is_initialized()
