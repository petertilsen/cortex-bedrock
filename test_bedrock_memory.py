#!/usr/bin/env python3
"""
Test script to verify Bedrock integration with in-memory ChromaDB
"""
import os
import sys
sys.path.append('.')

from cortex.memory_system import AgenticMemorySystem

def test_bedrock_with_inmemory_chroma():
    print("Testing Bedrock integration with in-memory ChromaDB...")
    
    try:
        # Initialize with Bedrock and in-memory ChromaDB
        memory = AgenticMemorySystem(
            llm_backend="bedrock",
            llm_model="anthropic.claude-3-haiku-20240307-v1:0",
            model_name="amazon.titan-embed-text-v1",
            enable_smart_collections=False,
            enable_background_processing=False,
            chroma_uri=None  # This should trigger in-memory mode
        )
        print("✅ Memory system initialized successfully")
        
        # Test adding a memory
        memory.add_note("User prefers morning meetings and uses VS Code")
        print("✅ Memory added successfully")
        
        # Test searching (this will test the full Bedrock integration)
        results = memory.search("What editor does the user like?")
        print(f"✅ Search completed. Found {len(results)} results")
        if results:
            print(f"   Result: {results[0]['content']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_bedrock_with_inmemory_chroma()
    sys.exit(0 if success else 1)
