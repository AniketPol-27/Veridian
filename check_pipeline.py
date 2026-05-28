import asyncio
from dotenv import load_dotenv
load_dotenv()

async def check():
    from backend.es.client import get_client, close_client
    es = get_client()
    
    # Check what the ingest pipeline looks like
    pipeline = await es.ingest.get_pipeline(id='veridian-elser-pipeline')
    print('Current pipeline definition:')
    import json
    print(json.dumps(pipeline.body, indent=2))
    
    # Check available ML models
    print('')
    print('Available ML models:')
    try:
        models = await es.ml.get_trained_models()
        for m in models['trained_model_configs']:
            print('  -', m['model_id'], '|', m.get('model_type', 'unknown'))
    except Exception as e:
        print('  Error fetching models:', str(e))
    
    await close_client()

asyncio.run(check())
