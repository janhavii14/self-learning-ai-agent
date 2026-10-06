from mem0 import Memory

# --------------------------------------------------------------
# Local Mem0 configuration
# --------------------------------------------------------------

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "my_memories",
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 768,
        },
    },
"llm": {
    "provider": "ollama",
    "config": {
        "model": "qwen3:4b",
        "temperature": 0,
        "max_tokens": 500,
        "ollama_base_url": "http://localhost:11434",
    },
},

    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
            "ollama_base_url": "http://localhost:11434",
        },
    },
}

m = Memory.from_config(config)


# --------------------------------------------------------------
# Message sequence
# --------------------------------------------------------------

messages = [
    {
        "role": "user",
        "content": "I'm planning to watch a movie tonight. Any recommendations?",
    },
    {
        "role": "assistant",
        "content": "How about a thriller movies? They can be quite engaging.",
    },
    {
        "role": "user",
        "content": "I'm not a big fan of thriller movies but I love sci-fi movies.",
    },
    {
        "role": "assistant",
        "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future.",
    },
]


# --------------------------------------------------------------
# Store inferred memories
# --------------------------------------------------------------

result = m.add(
    messages,
    user_id="default_user",
    metadata={"category": "movie_recommendations"},
)

print("Added memories:")
print(result)


# --------------------------------------------------------------
# Get all memories
# --------------------------------------------------------------

all_memories = m.get_all(user_id="default_user")

print("\nAll memories:")
print(all_memories)


# --------------------------------------------------------------
# Search for related memories
# --------------------------------------------------------------

related_memories = m.search(
    query="What do you know about me?",
    user_id="default_user",
)

print("\nRelated memories:")
print(related_memories)