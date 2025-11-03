#!/usr/bin/env python3
"""
Script to run pgvector migration
This enables the pgvector extension and adds vector columns to the database.
"""
import os
import sys
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import OperationalError

# Get database URL from environment or use default
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ai_platform")

def table_exists(conn, table_name):
    """Check if a table exists"""
    inspector = inspect(conn)
    return table_name in inspector.get_table_names()

def column_exists(conn, table_name, column_name):
    """Check if a column exists in a table"""
    if not table_exists(conn, table_name):
        return False
    inspector = inspect(conn)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns

def run_migration():
    """Run pgvector migration"""
    try:
        # Create engine
        engine = create_engine(DATABASE_URL)
        
        print("Connecting to database...")
        
        # Step 1: Enable pgvector extension (needs separate connection due to DDL)
        print("\n1. Enabling pgvector extension...")
        try:
            with engine.connect() as conn:
                # DDL operations need autocommit
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
                print("   ✅ pgvector extension enabled")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not enable pgvector extension: {e}")
            print("   Make sure pgvector is installed in PostgreSQL")
            return False
        
        # Verify extension
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT * FROM pg_extension WHERE extname = 'vector'"))
                if result.fetchone():
                    print("   ✅ pgvector extension verified")
                else:
                    print("   ❌ pgvector extension not found")
                    return False
        except Exception as e:
            print(f"   ⚠️  Could not verify extension: {e}")
            return False
        
        # Step 1.5: Create tables if they don't exist
        print("\n1.5. Creating tables if they don't exist...")
        try:
            # Add backend directory to path if needed
            import sys
            import os
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            if backend_dir not in sys.path:
                sys.path.insert(0, backend_dir)
            
            from config.database import Base
            from models import database_models
            # Create all tables defined in models
            Base.metadata.create_all(bind=engine)
            print("   ✅ Tables created/verified")
        except ImportError as e:
            print(f"   ⚠️  Warning: Could not import models: {e}")
            print("   Make sure you're running from the backend directory")
            print("   Tables will be created when you run main.py")
            # Continue anyway - we'll check for tables later
        except Exception as e:
            print(f"   ⚠️  Warning: Could not create tables automatically: {e}")
            print("   Tables may need to be created manually or via main.py")
            # Continue anyway - we'll check for tables later
            
        # Step 2: Add vector columns
        print("\n2. Adding vector columns...")
        with engine.connect() as conn:
            # Check if table exists first, then check/add column
            if table_exists(conn, 'training_data'):
                if column_exists(conn, 'training_data', 'embedding'):
                    print("   ⚠️  embedding column already exists in training_data")
                else:
                    try:
                        conn.execute(text("ALTER TABLE training_data ADD COLUMN embedding vector(384)"))
                        conn.commit()
                        print("   ✅ Added embedding column to training_data")
                    except Exception as e:
                        error_msg = str(e).lower()
                        # Check if column was added despite error or already exists
                        if column_exists(conn, 'training_data', 'embedding'):
                            print("   ✅ embedding column exists (may have been added already)")
                        elif 'already exists' in error_msg or 'duplicate' in error_msg:
                            print("   ⚠️  Column already exists")
                        else:
                            print(f"   ⚠️  Warning: Could not add embedding column: {e}")
                            conn.rollback()
            else:
                print("   ⚠️  training_data table does not exist yet")
                print("   Run 'python main.py' first to create tables, or create them manually")
            
            # Check if model_contexts table exists and add column
            if table_exists(conn, 'model_contexts'):
                if column_exists(conn, 'model_contexts', 'context_embedding_vector'):
                    print("   ⚠️  context_embedding_vector column already exists in model_contexts")
                else:
                    try:
                        conn.execute(text("ALTER TABLE model_contexts ADD COLUMN context_embedding_vector vector(384)"))
                        conn.commit()
                        print("   ✅ Added context_embedding_vector column to model_contexts")
                    except Exception as e:
                        error_msg = str(e).lower()
                        # Check if column was added despite error or already exists
                        if column_exists(conn, 'model_contexts', 'context_embedding_vector'):
                            print("   ✅ context_embedding_vector column exists (may have been added already)")
                        elif 'already exists' in error_msg or 'duplicate' in error_msg:
                            print("   ⚠️  Column already exists")
                        else:
                            print(f"   ⚠️  Warning: Could not add context_embedding_vector column: {e}")
                            conn.rollback()
            else:
                print("   ⚠️  model_contexts table does not exist yet")
                print("   Run 'python main.py' first to create tables, or create them manually")
            
        # Step 3: Create indexes (only if tables and columns exist)
        print("\n3. Creating vector indexes...")
        with engine.connect() as conn:
            if table_exists(conn, 'training_data') and column_exists(conn, 'training_data', 'embedding'):
                try:
                    # Check if index already exists
                    result = conn.execute(text("""
                        SELECT indexname FROM pg_indexes 
                        WHERE tablename = 'training_data' 
                          AND indexname = 'training_data_embedding_idx'
                    """))
                    if result.fetchone():
                        print("   ⚠️  Index training_data_embedding_idx already exists")
                    else:
                        conn.execute(text("""
                            CREATE INDEX training_data_embedding_idx 
                            ON training_data 
                            USING ivfflat (embedding vector_cosine_ops)
                            WITH (lists = 100)
                        """))
                        conn.commit()
                        print("   ✅ Created index on training_data.embedding")
                except Exception as e:
                    print(f"   ⚠️  Warning: Could not create index on training_data: {e}")
                    # Continue - index creation is not critical
            else:
                print("   ⚠️  Skipping index creation for training_data (table/column doesn't exist)")
            
            if table_exists(conn, 'model_contexts') and column_exists(conn, 'model_contexts', 'context_embedding_vector'):
                try:
                    # Check if index already exists
                    result = conn.execute(text("""
                        SELECT indexname FROM pg_indexes 
                        WHERE tablename = 'model_contexts' 
                          AND indexname = 'model_contexts_embedding_idx'
                    """))
                    if result.fetchone():
                        print("   ⚠️  Index model_contexts_embedding_idx already exists")
                    else:
                        conn.execute(text("""
                            CREATE INDEX model_contexts_embedding_idx 
                            ON model_contexts 
                            USING ivfflat (context_embedding_vector vector_cosine_ops)
                            WITH (lists = 100)
                        """))
                        conn.commit()
                        print("   ✅ Created index on model_contexts.context_embedding_vector")
                except Exception as e:
                    print(f"   ⚠️  Warning: Could not create index on model_contexts: {e}")
                    # Continue - index creation is not critical
            else:
                print("   ⚠️  Skipping index creation for model_contexts (table/column doesn't exist)")
            
            print("\n✅ Migration completed!")
            
            # Check if tables exist for final recommendation
            training_data_exists = table_exists(conn, 'training_data')
            model_contexts_exists = table_exists(conn, 'model_contexts')
            
            if not training_data_exists or not model_contexts_exists:
                print("\n⚠️  Note: Some tables don't exist yet.")
                print("   To create tables, run:")
                print("   python main.py")
                print("   (This will create all tables defined in your models)")
            
            print("\nNext steps:")
            print("1. If tables don't exist, run: python main.py (creates tables)")
            print("2. Install Python dependencies: pip install -r requirements.txt")
            print("3. Start your FastAPI server: python main.py")
            print("4. Train a langchain_rag model using the API")
            
            return True
            
    except OperationalError as e:
        print(f"\n❌ Database connection error: {e}")
        print(f"\nPlease check:")
        print(f"1. PostgreSQL is running")
        print(f"2. Database exists: {DATABASE_URL.split('/')[-1]}")
        print(f"3. Credentials are correct")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1)

