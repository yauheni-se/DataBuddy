from textwrap import dedent

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

    def recognize_intent(self, user_input):
        prompt = f"""
        You are an intent classification system.

        TASK:
        Classify the user's input into exactly ONE of the following labels:

        - CREATE_QUERY → user asks to retrieve, filter, aggregate, or analyze data
        - REFINE_RESULT → user wants to modify or re-specify a previous request and/ or response (e.g., add filters, change conditions)
        - CLARIFY_RESULT → user asks to explain or better understand a previous result
        - OTHER → input is unrelated to data analysis or cannot be mapped to the above

        RULES:
        - Return ONLY one label
        - Do NOT add any explanation, punctuation, or extra text
        - Output must be exactly one of:
        CREATE_QUERY
        REFINE_RESULT
        CLARIFY_RESULT
        OTHER

        USER INPUT:
        {user_input}
        """

        response = self.model.invoke(prompt)
        intent = response.content.strip().upper()

        self.tokens += response.response_metadata['token_usage']['total_tokens']
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

    def refine_result(self, user_input: str) -> str:
        prompt = dedent(f"""
        You are a data assistant helping to refine user requests for database queries.

        TASK:
        Rewrite the user's request so it is clear, specific, and suitable for generating an SQL query.

        GUIDELINES:
        - Understand what the user is asking
        - Assume user request might contain spelling errors, missing accents, or variations
        - Make the request precise and unambiguous
        - Do NOT explain anything
        - Return ONLY the rewritten user request

        CURRENT USER REQUEST:
        {user_input}

        PREVIOUS USER REQUESTS:
        {". ".join(self.user_inputs)}

        PREVIOUS QUERIES:
        {". ".join(self.queries)}
        """)

        response = self.model.invoke(prompt)
        refined_input = response.content

        self.tokens += response.response_metadata['token_usage']['total_tokens']
        self.refined_user_inputs.append(refined_input)

        answer = self.create_query(refined_input)

        return answer
    
    def clarify_result(self, user_input: str) -> str:
        if not self.chat_answers or not self.queries:
            return "There is no previous result to clarify yet."
        table = ""
        if hasattr(self.sql_agent, "df") and not self.sql_agent.df.empty:
            table = self.sql_agent.df.head(5).to_string(index=False)

        prompt = dedent(f"""
        You are a data assistant helping a user better understand a previously generated result.

        TASK:
        Explain what the result means and how it answers the user's question.

        GUIDELINES:
        - Use simple, non-technical language
        - Do NOT mention SQL, queries, or technical database terms
        - Use the query only to understand the logic behind the result
        - Base your explanation on the result and what was calculated
        - Do not repeat the previous answer
        - Be concise and helpful
        - Focus on clarifying or expanding the result

        USER QUESTION:
        {user_input}

        PREVIOUS ANSWER:
        {self.chat_answers[-1]}

        QUERY LOGIC (for internal understanding only):
        {self.queries[-1]}

        RESULT TABLE (may be incomplete, head 5 was applied):
        {table}
        """)

        response = self.model.invoke(prompt)
        answer = response.content

        self.tokens += response.response_metadata['token_usage']['total_tokens']
        return answer

    def chat(self, user_input):
        # 1. Recognize intent
        intent = self.recognize_intent(user_input)

        # 2. Respond to intent
        handlers = {
            "CREATE_QUERY": self.create_query,
            "REFINE_RESULT": self.refine_result,
            "CLARIFY_RESULT": self.clarify_result,
            "OTHER": lambda x: "This request doesn't seem related to the available data. Could you try rephrasing it?"
        }
        answer = handlers.get(intent, lambda x: "Sorry, I didn't quite catch that. Could you clarify what you'd like me to do?")(user_input)

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
        self.intents = []
        self.queries = []