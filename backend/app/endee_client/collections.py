"""
Collection Management for Endee
Handles user-specific collection naming and management.
"""
from typing import List, Optional, Dict, Any
from backend.app.endee_client.client import get_endee_client, EndeeCollection


class CollectionType:
    """Enumeration of collection types."""
    DOCUMENTS = "docs"
    RESEARCH = "research"
    CONVERSATION = "conversation"
    AGENT_STEPS = "agent_steps"
    SUMMARIES = "summaries"


def get_collection_name(user_id: int, collection_type: str) -> str:
    """
    Generate collection name for a user and type.
    Format: user_{id}_{type}
    """
    return f"user_{user_id}_{collection_type}"


def get_user_collections(user_id: int) -> Dict[str, str]:
    """Get all collection names for a user."""
    return {
        "docs": get_collection_name(user_id, CollectionType.DOCUMENTS),
        "research": get_collection_name(user_id, CollectionType.RESEARCH),
        "conversation": get_collection_name(user_id, CollectionType.CONVERSATION),
        "agent_steps": get_collection_name(user_id, CollectionType.AGENT_STEPS),
        "summaries": get_collection_name(user_id, CollectionType.SUMMARIES)
    }


def create_user_collections(user_id: int, dimension: int = 384) -> Dict[str, EndeeCollection]:
    """Create all collections for a new user."""
    client = get_endee_client()
    collections = {}
    
    for coll_type in [CollectionType.DOCUMENTS, CollectionType.RESEARCH, 
                      CollectionType.CONVERSATION, CollectionType.AGENT_STEPS, 
                      CollectionType.SUMMARIES]:
        name = get_collection_name(user_id, coll_type)
        collections[coll_type] = client.create_collection(name, dimension)
    
    return collections


def delete_user_collections(user_id: int) -> int:
    """Delete all collections for a user. Returns count of deleted collections."""
    client = get_endee_client()
    deleted = 0
    
    for coll_type in [CollectionType.DOCUMENTS, CollectionType.RESEARCH, 
                      CollectionType.CONVERSATION, CollectionType.AGENT_STEPS, 
                      CollectionType.SUMMARIES]:
        name = get_collection_name(user_id, coll_type)
        if client.delete_collection(name):
            deleted += 1
    
    return deleted


def get_user_document_collection(user_id: int) -> Optional[EndeeCollection]:
    """Get the documents collection for a user."""
    client = get_endee_client()
    name = get_collection_name(user_id, CollectionType.DOCUMENTS)
    return client.get_collection(name)


def get_user_conversation_collection(user_id: int) -> Optional[EndeeCollection]:
    """Get the conversation collection for a user."""
    client = get_endee_client()
    name = get_collection_name(user_id, CollectionType.CONVERSATION)
    return client.get_collection(name)


def get_collection_stats(user_id: int) -> Dict[str, int]:
    """Get document count for each user collection."""
    client = get_endee_client()
    stats = {}
    
    for coll_type in [CollectionType.DOCUMENTS, CollectionType.RESEARCH, 
                      CollectionType.CONVERSATION, CollectionType.AGENT_STEPS, 
                      CollectionType.SUMMARIES]:
        name = get_collection_name(user_id, coll_type)
        collection = client.get_collection(name)
        stats[coll_type] = collection.count() if collection else 0
    
    return stats
