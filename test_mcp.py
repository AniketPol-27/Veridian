import asyncio
from dotenv import load_dotenv
load_dotenv()

async def test_mcp_tools():
    from backend.mcp.client import VeridianMCPClient
    from backend.es.client import close_client

    mcp = VeridianMCPClient()
    passed = 0
    failed = 0

    print()
    print('=' * 60)
    print('  VERIDIAN MCP Tools Test')
    print('=' * 60)

    # TOOL 1: find_related_signals
    print()
    print('TOOL 1: find_related_signals')
    print('-' * 40)
    try:
        result = await mcp.find_related_signals(
            query='OpenAI artificial intelligence product launch',
            time_range_days=30,
            min_score=0.5,
            limit=5,
        )
        print('  Signals found:', result.total_found)
        print('  Returned:', len(result.signals))
        print('  Strategy:', result.search_strategy)
        if result.signals:
            s = result.signals[0]
            print('  Top signal:', s.signal_id, '|', s.signal_type)
            print('  Relevance:', s.relevance_score)
        print('  PASS')
        passed += 1
    except Exception as e:
        print('  FAIL:', str(e))
        failed += 1

    # TOOL 2: compute_entity_anomaly
    print()
    print('TOOL 2: compute_entity_anomaly')
    print('-' * 40)
    try:
        result = await mcp.compute_entity_anomaly(
            entity_name='OpenAI',
            metric='impact_score',
            lookback_days=30,
            current_window_days=7,
        )
        print('  Entity:', result.entity_name)
        print('  Is anomalous:', result.is_anomalous)
        print('  Z-score:', result.z_score)
        print('  Direction:', result.direction)
        print('  Confidence:', result.confidence)
        print('  PASS')
        passed += 1
    except Exception as e:
        print('  FAIL:', str(e))
        failed += 1

    # TOOL 3: find_historical_pattern
    print()
    print('TOOL 3: find_historical_pattern')
    print('-' * 40)
    try:
        from backend.es.client import get_client
        es = get_client()
        recent = await es.search(
            index='veridian_signals',
            body={'query': {'match_all': {}}, 'size': 3, '_source': ['signal_id']}
        )
        signal_ids = [h['_source']['signal_id'] for h in recent['hits']['hits']]

        result = await mcp.find_historical_pattern(
            current_signal_ids=signal_ids,
            top_k=3,
        )
        print('  Signals used:', result.fingerprint_signal_count)
        print('  Matches found:', len(result.matches))
        print('  Total searched:', result.total_searched)
        print('  Has strong precedent:', result.has_strong_precedent)
        print('  Note: 0 matches expected (no validated chains yet)')
        print('  PASS')
        passed += 1
    except Exception as e:
        print('  FAIL:', str(e))
        failed += 1

    # TOOL 4: get_entity_state
    print()
    print('TOOL 4: get_entity_state')
    print('-' * 40)
    try:
        result = await mcp.get_entity_state(
            entity_name='OpenAI',
            dimensions=['news', 'community', 'financial'],
            lookback_days=30,
        )
        print('  Entity:', result.entity_name)
        print('  Total signals:', result.total_signal_count)
        print('  Risk score:', result.overall_risk_score)
        print('  Opportunity score:', result.overall_opportunity_score)
        for dim, state in result.dimension_states.items():
            print(f'  {dim}: {state.signal_count} signals | trend: {state.trend}')
        print('  PASS')
        passed += 1
    except Exception as e:
        print('  FAIL:', str(e))
        failed += 1

    # TOOL 5: store_prediction
    print()
    print('TOOL 5: store_prediction')
    print('-' * 40)
    try:
        from backend.mcp.schemas import PredictionType
        result = await mcp.store_prediction(
            chain_id='VRD-CHN-TEST0001',
            prediction_text='OpenAI will launch a new enterprise product within 60 days based on recent hiring signals and partnership announcements.',
            prediction_type=PredictionType.COMPETITIVE,
            confidence=0.65,
            signal_ids=signal_ids[:2],
            validate_in_days=60,
            key_assumptions=['Hiring signals indicate product readiness', 'Partnership suggests go-to-market preparation'],
            stakes_level=2,
        )
        print('  Prediction ID:', result.prediction_id)
        print('  Validate at:', result.validate_at)
        print('  Success:', result.success)
        print('  Message:', result.message)
        print('  PASS')
        passed += 1
    except Exception as e:
        print('  FAIL:', str(e))
        failed += 1

    # TOOL 6: run_calibration_cycle
    print()
    print('TOOL 6: run_calibration_cycle')
    print('-' * 40)
    try:
        result = await mcp.run_calibration_cycle()
        print('  Calibration ID:', result.calibration_id)
        print('  Evaluated:', result.predictions_evaluated)
        print('  Overall accuracy:', result.overall_accuracy)
        print('  Success:', result.success)
        print('  Report:', result.report_text[:100])
        print('  PASS')
        passed += 1
    except Exception as e:
        print('  FAIL:', str(e))
        failed += 1

    print()
    print('=' * 60)
    print(f'  Results: {passed} PASSED | {failed} FAILED')
    print('=' * 60)
    print()

    await close_client()

asyncio.run(test_mcp_tools())
