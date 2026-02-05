"""
Database Initialization Script
Initializes the SQLite database with schema.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.app.config import config
from backend.app.database.connection import init_database, drop_database


def main():
    """Initialize the database."""
    print("=" * 50)
    print("Agentic RAG Research Assistant")
    print("Database Initialization Script")
    print("=" * 50)
    
    # Ensure directories exist
    config.ensure_directories()
    
    # Ask user
    print(f"\nDatabase path: {config.DATABASE_URL}")
    
    if "--reset" in sys.argv:
        print("\n⚠️  WARNING: This will delete all existing data!")
        confirm = input("Type 'yes' to confirm database reset: ")
        
        if confirm.lower() == "yes":
            print("\nDropping existing database...")
            drop_database()
            print("Creating new database...")
            init_database()
            print("\n✅ Database reset complete!")
        else:
            print("\n❌ Reset cancelled.")
    else:
        print("\nInitializing database...")
        init_database()
        print("\n✅ Database initialization complete!")
    
    print("\nYou can now run the application with:")
    print("  streamlit run frontend/streamlit_app.py")
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
