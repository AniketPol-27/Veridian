import asyncio
from dotenv import load_dotenv
load_dotenv()

async def verify():
    from backend.es.client import get_client, close_client
    es = get_client()
    
    count = await es.count(index='veridian_signals')
    print('Total signals in Elasticsearch:', count['count'])
    
    result = await es.search(
        index='veridian_signals',
        body={
            'size': 1,
            'query': {'match_all': {}},
            'sort': [{'ingested_at': 'desc'}]
        }
    )
    
    hit = result['hits']['hits'][0]['_source']
    print('')
    print('Latest signal:')
    print('  ID:          ', hit['signal_id'])
    print('  Source:      ', hit['source'])
    print('  Type:        ', hit['signal_type'])
    print('  Entities:    ', hit['entity_hints'])
    print('  Urgency:     ', hit['urgency_score'])
    print('  Impact:      ', hit['impact_score'])
    print('  Has ELSER:   ', 'semantic_embedding' in hit)
    print('  Content:     ', hit['raw_content'][:80])
    
    await close_client()

asyncio.run(verify())
