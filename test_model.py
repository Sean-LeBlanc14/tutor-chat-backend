# test_model.py
from query_bot import llama_service

print("Testing model loading and generation...")
try:
    print(f"Using vLLM: {llama_service.use_vllm}")
    
    # Test a simple generation
    response = ""
    for token in llama_service.generate_stream("What is psychology?"):
        response += token
        print(token, end="", flush=True)
    
    print(f"\n\nSuccess! Full response: {response}")
    
except Exception as e:
    print(f"Model test failed: {e}")
    import traceback
    traceback.print_exc()
