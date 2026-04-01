from textwrap import dedent
from utils import read_prompt

class DataBuddyAgent:
    def __init__(self, model, sql_agent, chat_history=5):
        self.model = model
        self.sql_agent = sql_agent
        self.chat_history = chat_history

        self.tokens = 0
        self.user_inputs = []
        self.chat_answers = []
        self.queries = []
        self.intents = []
        self.refined_user_inputs = []

    def __str__(self):
        return (
            f"User inputs: {self.user_inputs}\n"
            f"Recognized intents: {self.intents}\n"
            f"Buddy answers: {self.chat_answers}\n"
            f"Generated queries: {self.queries}\n"
            f"Refined user inputs: {self.refined_user_inputs}\n"
            f"Total tokens spent: {self.tokens}"
        )

    def recognize_intent(self, user_input):
        prompt = read_prompt(
            "prompts/orchestrator/recognize_intent.txt",
            user_input=user_input,
            user_inputs=". ".join(self.user_inputs)
        )

        response = self.model.invoke(prompt)
        intent = response.text.strip().upper()

        self.tokens += response.usage_metadata['total_tokens']
        self.intents.append(intent)
        return intent

    def create_query(self, user_input: str) -> str:
        answer = self.sql_agent.chat(user_input)
        query = self.sql_agent.query

        self.tokens += self.sql_agent.tokens
        if self.sql_agent.failed:
            return answer
        self.queries.append(query)
        return answer

    def refine_intent(self, user_input: str) -> str:
        prompt = read_prompt(
            "prompts/orchestrator/refine_intent.txt",
            user_input=user_input,
            user_inputs=". ".join(self.user_inputs),
            queries=". ".join(self.queries)
        )

        response = self.model.invoke(prompt)
        refined_input = response.text

        self.tokens += response.usage_metadata['total_tokens']
        self.refined_user_inputs.append(refined_input)

        answer = self.create_query(refined_input)

        return answer
    
    def clarify_result(self, user_input: str) -> str:
        if not self.chat_answers or not self.queries:
            return "There is no previous result to clarify yet."
        table = ""
        if hasattr(self.sql_agent, "df") and not self.sql_agent.df.empty:
            table = self.sql_agent.df.head(5).to_string(index=False)

        prompt = read_prompt(
            "prompts/orchestrator/clarify_result.txt",
            user_input=user_input,
            answer=self.chat_answers[-1],
            query=self.queries[-1],
            table=table
        )

        response = self.model.invoke(prompt)
        answer = response.text

        self.tokens += response.usage_metadata['total_tokens']
        return answer

    def redirect_intent(self, user_input) -> str:
        prompt = read_prompt(
            "prompts/orchestrator/redirect_intent.txt",
            user_input=user_input
        )

        response = self.model.invoke(prompt)
        answer = response.text
        self.tokens += response.usage_metadata['total_tokens']
        return answer
    
    def clarify_intent(self, user_input) -> str:
        prompt = read_prompt(
            "prompts/orchestrator/clarify_intent.txt",
            user_input=user_input
        )

        response = self.model.invoke(prompt)
        answer = response.text
        self.tokens += response.usage_metadata['total_tokens']
        return answer

    def chat(self, user_input):
        # 1. Recognize intent
        intent = self.recognize_intent(user_input)

        # 2. Respond to intent
        handlers = {
            "CREATE_NEW_QUERY": self.create_query,
            "REFINE_PREV_QUERY": self.refine_intent,
            "CLARIFY_RESULT": self.clarify_result,
            "OTHER": self.redirect_intent
        }
        answer = handlers.get(intent, self.clarify_intent)(user_input)

        # 3. Manage memory
        self.user_inputs.append(user_input)
        self.chat_answers.append(answer)

        self.user_inputs = self.user_inputs[-self.chat_history:]
        self.chat_answers = self.chat_answers[-self.chat_history:]
        self.queries = self.queries[-self.chat_history:]
        self.intents = self.intents[-self.chat_history:]
        self.refined_user_inputs = self.refined_user_inputs[-self.chat_history:]

        # 4. Return answer
        output_dict = {
            "intent": intent,
            "answer": answer,
            "last query": self.queries[-1] if self.queries else "",
            "token spent": self.tokens
        }
        return output_dict

    def end(self):
        self.tokens = 0
        self.user_inputs = []
        self.chat_answers = []
        self.queries = []
        self.intents = []
        self.refined_user_inputs = []