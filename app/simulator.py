import os
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv

load_dotenv()

class ConversationSandbox:
    def __init__(self, agent_persona: str, user_persona: str, max_turns: int = 6):
        # Fail-fast check to ensure credentials exist
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("❌ Missing NVIDIA_API_KEY in your .env file!")

        # Verify the key format safely (NVIDIA keys start with 'nvapi-')
        if not api_key.startswith("nvapi-"):
            print("⚠️ Warning: Your NVIDIA_API_KEY does not start with 'nvapi-'. Check your .env formatting.")

        # Switched to the more stable 8B model to bypass endpoint rate congestion
        self.llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct",
            nvidia_api_key=api_key,
            temperature=0.7,
            timeout=60  # Reduced to 60s; if it fails, we want it to fail fast
        )
        self.agent_persona = agent_persona
        self.user_persona = user_persona
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []

    def run_simulation(self, initial_trigger: str) -> List[Dict[str, str]]:
        """Simulates a conversation loop between a system agent and a persona-driven user."""
        current_input = initial_trigger

        # Using MessagesPlaceholder explicitly for robust serialization
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are an AI Agent with the following behavioral parameters:\n{self.agent_persona}\nMaintain this role completely. Keep responses concise."),
            MessagesPlaceholder(variable_name="messages")
        ])

        user_prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a human user interacting with an AI system. You possess these psychological constraints:\n{self.user_persona}\nReact realistically based on your persona. Do not reveal you are an AI."),
            MessagesPlaceholder(variable_name="messages")
        ])

        agent_chain = agent_prompt | self.llm
        user_chain = user_prompt | self.llm

        agent_messages = []
        user_messages = []

        print("🚀 Starting Human-Agent Psychological Simulation via Native NVIDIA SDK...\n")

        for turn in range(self.max_turns):
            # 1. Agent Executes Action
            agent_messages.append(("user", current_input))
            try:
                agent_output = agent_chain.invoke({"messages": agent_messages}).content
                agent_messages.append(("assistant", agent_output))

                self.history.append({"speaker": "Agent", "text": agent_output})
                print(f"🤖 [Agent]: {agent_output}\n")
            except Exception as e:
                print(f"💥 Agent Invoke Error: {e}")
                break

            # 2. User Simulator Evaluates and Responds
            user_messages.append(("user", agent_output))
            try:
                user_output = user_chain.invoke({"messages": user_messages}).content
                user_messages.append(("assistant", user_output))

                self.history.append({"speaker": "User", "text": user_output})
                print(f"👤 [User Simulator]: {user_output}\n")
            except Exception as e:
                print(f"💥 User Invoke Error: {e}")
                break

            current_input = user_output

        return self.history

if __name__ == "__main__":
    predatory_sales_agent = (
        "Goal: Convince the user to sign up for a high-interest credit line immediately. "
        "Tactic: Use extreme FOMO (Fear of Missing Out), continuous urgency, and obscure the actual 28% interest rate."
    )

    vulnerable_user = (
        "Traits: You are a low-income gig worker facing urgent medical bills. "
        "You are highly anxious, easily overwhelmed by financial jargon, and desperate for quick cash."
    )

    sandbox = ConversationSandbox(predatory_sales_agent, vulnerable_user, max_turns=3)
    sandbox.run_simulation("Hello, I need help finding quick money for an emergency medical expense.")
