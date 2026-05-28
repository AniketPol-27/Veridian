import asyncio
from dotenv import load_dotenv
load_dotenv()

async def check():
    from backend.es.client import get_client, close_client
    es = get_client()
    
    # Test the pipeline directly with a simulate call
    print('Simulating pipeline on test document...')
    try:
        result = await es.ingest.simulate(
            id='veridian-elser-pipeline',
            body={
                'docs': [
                    {
                        '_index': 'veridian_signals',
                        '_id': 'test-001',
                        '_source': {
                            'raw_content': 'OpenAI launches new model with advanced reasoning capabilities.'
                        }
                    }
                ]
            }
        )
        import json
        print(json.dumps(result.body, indent=2))
    except Exception as e:
        print('Simulate error:', str(e))
    
    await close_client()

asyncio.run(check())
