import redis
import json
import hashlib
import os
from functools import wraps
from typing import Optional, Callable, Any

# Redis connection
redis_client: Optional[redis.Redis] = None

def init_redis(host: str = None, port: int = None, db: int = 0) -> redis.Redis:
    """Initialize Redis connection"""
    global redis_client
    
    host = host or os.getenv('REDIS_HOST', 'localhost')
    port = port or int(os.getenv('REDIS_PORT', '6379'))
    
    try:
        redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            socket_connect_timeout=5
        )
        # Test connection
        redis_client.ping()
        print(f"✓ Redis connected successfully at {host}:{port}")
        return redis_client
    except Exception as e:
        print(f"⚠ Redis connection failed: {e}. Caching will be disabled.")
        redis_client = None
        return None

def get_redis_client() -> Optional[redis.Redis]:
    """Get the Redis client instance"""
    return redis_client

def generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """Generate a unique cache key from arguments"""
    # Create a string representation of all arguments
    key_data = f"{prefix}:{str(args)}:{str(sorted(kwargs.items()))}"
    # Hash to create a fixed-length key
    return f"{prefix}:{hashlib.md5(key_data.encode()).hexdigest()}"

def cache_result(prefix: str = "cache", ttl: int = 3600):
    """Decorator to cache function results in Redis
    
    Args:
        prefix: Prefix for the cache key
        ttl: Time to live in seconds (default: 1 hour)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if redis_client is None:
                # If Redis is not available, just call the function
                return await func(*args, **kwargs)
            
            # Generate cache key
            cache_key = generate_cache_key(prefix, *args, **kwargs)
            
            try:
                # Try to get from cache
                cached_value = redis_client.get(cache_key)
                if cached_value:
                    print(f"✓ Cache hit for {prefix}")
                    return json.loads(cached_value)
            except Exception as e:
                print(f"⚠ Cache read error: {e}")
            
            # Call the actual function
            result = await func(*args, **kwargs)
            
            try:
                # Store in cache
                redis_client.setex(cache_key, ttl, json.dumps(result))
                print(f"✓ Cache stored for {prefix}")
            except Exception as e:
                print(f"⚠ Cache write error: {e}")
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if redis_client is None:
                return func(*args, **kwargs)
            
            cache_key = generate_cache_key(prefix, *args, **kwargs)
            
            try:
                cached_value = redis_client.get(cache_key)
                if cached_value:
                    print(f"✓ Cache hit for {prefix}")
                    return json.loads(cached_value)
            except Exception as e:
                print(f"⚠ Cache read error: {e}")
            
            result = func(*args, **kwargs)
            
            try:
                redis_client.setex(cache_key, ttl, json.dumps(result))
                print(f"✓ Cache stored for {prefix}")
            except Exception as e:
                print(f"⚠ Cache write error: {e}")
            
            return result
        
        # Return appropriate wrapper based on whether function is async
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

def clear_cache_pattern(pattern: str = "*"):
    """Clear cache entries matching a pattern"""
    if redis_client is None:
        return 0
    
    try:
        keys = redis_client.keys(pattern)
        if keys:
            return redis_client.delete(*keys)
        return 0
    except Exception as e:
        print(f"⚠ Cache clear error: {e}")
        return 0

def get_cache_stats() -> dict:
    """Get cache statistics"""
    if redis_client is None:
        return {"status": "disabled"}
    
    try:
        info = redis_client.info()
        return {
            "status": "connected",
            "keys": redis_client.dbsize(),
            "memory_used": info.get('used_memory_human'),
            "hits": info.get('keyspace_hits', 0),
            "misses": info.get('keyspace_misses', 0)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
