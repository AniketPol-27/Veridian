import asyncio
from dotenv import load_dotenv
load_dotenv()

async def check():
    from backend.es.client import get_client, close_client
    es = get_client()
    
    # Count docs WITH semantic_embedding
    with_embedding = await es.count(
        index='veridian_signals',
        body={'query': {'exists': {'field': 'semantic_embedding'}}}
    )
    
    # Count docs WITHOUT semantic_embedding  
    without_embedding = await es.count(
        index='veridian_signals',
        body={'query': {'bool': {'must_not': {'exists': {'field': 'semantic_embedding'}}}}}
    )
    
    total = await es.count(index='veridian_signals')
    
    print('Total documents:        ', total['count'])
    print('With ELSER embedding:   ', with_embedding['count'])
    print('Without ELSER embedding:', without_embedding['count'])
    
    await close_client()

asyncio.run(check())
