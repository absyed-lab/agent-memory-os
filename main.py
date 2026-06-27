import os
import json
import re
from agent.support_agent import CustomerSupportAgent

def sanitize_id(customer_id: str) -> str:
    sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', customer_id)
    if not sanitized or not sanitized[0].isalnum():
        sanitized = "c_" + sanitized
    if len(sanitized) < 3:
        sanitized = sanitized + "_id"
    return sanitized[:512]

def main():
    print("\n" + "="*50)
    print("  Agent Memory OS - Customer Support Demo")
    print("  Model: llama3.1:8b (running locally)")
    print("="*50)

    raw_id = input("\nEnter customer ID (or press Enter for demo_user): ").strip()
    if not raw_id:
        raw_id = "demo_user"
    customer_id = sanitize_id(raw_id)

    os.makedirs("./data", exist_ok=True)
    agent = CustomerSupportAgent(customer_id=customer_id)
    agent.remember_fact(f"Customer {customer_id} is on the Pro plan", importance=0.9)
    agent.remember_fact(f"Customer {customer_id} prefers concise responses", importance=0.7)

    # Load past context and greet with memory
    greeting_context = agent.get_greeting_context()
    if greeting_context:
        print(f"\n[Memory loaded: past episodes found for {customer_id}]")
        opening = agent.chat(f"The customer just reconnected. Greet them warmly and briefly reference their last issue. Context: {greeting_context}")
        print(f"Agent: {opening}\n")
    else:
        print(f"\nAgent ready. New customer: {customer_id}")
        print("No past history found.\n")

    print("Commands: quit | status | end\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "status":
            print("\nMEMORY STATUS:")
            print(json.dumps(agent.memory_status(), indent=2))
            print()
            continue
        if user_input.lower() == "end":
            print("\n[Ending conversation and storing episode...]")
            summary = agent.end_conversation()
            print(f"Summary stored:\n{summary}\n")
            continue
        print("Agent thinking...   ", end="\r")
        response = agent.chat(user_input)
        print(f"Agent: {response}\n")

if __name__ == "__main__":
    main()
