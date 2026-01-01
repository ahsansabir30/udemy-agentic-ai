"""
Simple knowledge base search tool (fallback without ChromaDB).
"""

import os
from typing import List, Dict, Any
from pathlib import Path
import sys
import re

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.models import udahub
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

class KnowledgeBaseSearch:
    """Simple text-based search for knowledge base."""

    def __init__(self, db_path: str = "data/core/udahub.db"):
        self.db_path = db_path

    def get_db_session(self):
        """Get database session."""
        engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        Session = sessionmaker(bind=engine)
        return Session()

    def search_knowledge(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Search the knowledge base using simple text matching."""
        with self.get_db_session() as session:
            articles = session.query(udahub.Knowledge).filter(
                udahub.Knowledge.account_id == "cultpass"
            ).all()

            # Simple relevance scoring
            scored_articles = []
            query_lower = query.lower()

            for article in articles:
                title_lower = article.title.lower()
                content_lower = article.content.lower()
                tags_lower = (article.tags or "").lower()

                # Calculate relevance score
                score = 0
                if query_lower in title_lower:
                    score += 10
                if query_lower in content_lower:
                    score += 5
                if any(word in tags_lower for word in query_lower.split()):
                    score += 3

                # Word matching
                query_words = set(query_lower.split())
                title_words = set(title_lower.split())
                content_words = set(content_lower.split())
                tags_words = set(tags_lower.split())

                word_matches = len(query_words.intersection(title_words)) * 2 + \
                              len(query_words.intersection(content_words)) + \
                              len(query_words.intersection(tags_words))

                score += word_matches

                if score > 0:
                    scored_articles.append((score, article))

            # Sort by score and return top results
            scored_articles.sort(key=lambda x: x[0], reverse=True)
            results = scored_articles[:n_results]

            search_results = []
            for score, article in results:
                search_results.append({
                    "article_id": article.article_id,
                    "title": article.title,
                    "content": article.content,
                    "tags": article.tags,
                    "relevance_score": score
                })

            return search_results

# Tool function for LangChain
def search_knowledge_base(query: str, n_results: int = 3) -> Dict[str, Any]:
    """Search the knowledge base using simple text matching.

    Args:
        query: The search query
        n_results: Number of results to return

    Returns:
        Dictionary with response text and confidence information
    """
    search = KnowledgeBaseSearch()
    results = search.search_knowledge(query, n_results)

    if not results:
        return {
            "response": "No relevant information found in the knowledge base.",
            "confidence": 0.0,
            "found_results": False
        }

    # Calculate overall confidence based on top result score
    # Normalize score to 0-1 range (assuming max possible score ~50)
    top_score = results[0]['relevance_score']
    confidence = min(top_score / 20.0, 1.0)  # Cap at 1.0

    formatted_results = []
    for result in results:
        formatted_results.append(f"""
**{result['title']}**
{result['content'][:500]}{'...' if len(result['content']) > 500 else ''}
*Tags: {result['tags']}*
*Relevance: {result['relevance_score']}*
""")

    return {
        "response": "\n---\n".join(formatted_results),
        "confidence": confidence,
        "found_results": True,
        "top_score": top_score
    }

if __name__ == "__main__":
    # Test the search
    results = search_knowledge_base("login")
    print(results)