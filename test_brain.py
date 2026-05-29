import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def test_veridian_reasoning():
    from backend.engines.causal_chain_engine import CausalChainEngine

    engine = CausalChainEngine()

    trigger_id = "VRD-SIG-TEST-001"
    trigger_content = "OpenAI is reportedly hiring a massive team of senior platform engineers and Kubernetes experts for a new enterprise infrastructure initiative. Multiple job postings appeared simultaneously suggesting a coordinated push into enterprise infrastructure."
    trigger_entities = ["OpenAI"]

    print()
    print("=" * 60)
    print("  VERIDIAN: CAUSAL CHAIN PROTOCOL INITIATED")
    print("=" * 60)
    print(f"  Trigger: {trigger_content[:80]}...")
    print(f"  Entity:  {trigger_entities[0]}")
    print()

    try:
        chain = await engine.build_chain(
            trigger_signal_id=trigger_id,
            trigger_content=trigger_content,
            trigger_entities=trigger_entities,
        )

        if chain:
            print()
            print("=" * 60)
            print("  CAUSAL CHAIN CONSTRUCTED")
            print("=" * 60)
            print()
            print(chain.to_summary())
            print()
            print("KEY ASSUMPTIONS:")
            for a in chain.key_assumptions:
                print(f"  • {a}")
            print()
            print(f"AUDIT TRAIL: {len(chain.reasoning_steps)} steps logged")
            for step in chain.reasoning_steps:
                print(f"  Step {step['step_number']}: {step['step_name']}")
                print(f"    Tool: {step['tool_used']}")
                print(f"    Output: {step['output_summary'][:100]}")
        else:
            print()
            print("  No chain constructed.")
            print("  Signal was filtered as noise or no hypothesis survived.")

    except Exception as e:
        import traceback
        print(f"  ERROR: {str(e)}")
        traceback.print_exc()

    from backend.es.client import close_client
    await close_client()

asyncio.run(test_veridian_reasoning())
