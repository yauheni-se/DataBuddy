import sqlite3
import re
import pandas as pd
from textwrap import dedent
from IPython.display import display

def read_prompt(path: str, **kwargs) -> str:
    """
    Reads a prompt template from a text file and formats it with the given keyword arguments.

    Args:
        path (str): Path to the prompt .txt file.
        **kwargs: Key-value pairs to fill placeholders in the template.

    Returns:
        str: Formatted prompt string.
    """
    with open(path, "r", encoding="utf-8") as f:
        template = f.read()
    return template.format(**kwargs)


class SQLAgent:
    def __init__(self, model, db_name: str, db_description: str, max_retries=3):
        self.model = model
        self.db_name = db_name
        self.db_description = db_description
        self.max_retries = max_retries

        # States redefined with every .run execution implicitly
        self.query = ""

        # States redefined with every .run execution explicitly
        self.failed = False
        self.previous_queries = []
        self.errors = []
        self.attempt = 0
        self.tokens = 0
        self.df = pd.DataFrame()

    def generate_query(self):
        # 1. List interpretations
        prompt = read_prompt("prompts\sql_agent\list_interpretations.txt", db_description=self.db_description, user_input=self.user_input)
        messages = [
            {"role": "system", "content": "You are an expert data analyst and intent recognition system."},
            {"role": "user", "content": prompt}
        ]
        response1 = self.model.invoke(messages)
        messages.append({"role": "assistant", "content": response1.content})

        # 2. Evaluate interpretations
        prompt = read_prompt("prompts\sql_agent\eval_interpratetions.txt")
        messages.append({"role": "user", "content": prompt})
        response2 = self.model.invoke(messages)
        messages.append({"role": "assistant", "content": response2.content})

        # 3. Create query
        prompt = read_prompt("prompts\sql_agent\create_query.txt")
        messages.append({"role": "user", "content": prompt})
        response3 = self.model.invoke(messages)
        self.query = response3.content

        # 4. Update attributes
        self.previous_queries.append(self.query)
        self.attempt += 1
        self.tokens += (
            response1.response_metadata['token_usage']['total_tokens'] + 
            response2.response_metadata['token_usage']['total_tokens'] + 
            response3.response_metadata['token_usage']['total_tokens']
        )
        return self.query

    def refine_query(self):
        prompt = read_prompt(
            "prompts\sql_agent\refine_query.txt", 
            db_description=self.db_description, 
            user_input=self.user_input,
            previous_queries=", ".join(self.previous_queries),
            errors=", ".join(self.errors)
        )

        response = self.model.invoke(prompt)
        self.query = response.content
        self.previous_queries.append(self.query)
        self.attempt += 1
        self.tokens += response.response_metadata['token_usage']['total_tokens']
        return self.query

    def check_query(self):
        query = self.query.strip().upper()
        if not query.startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed.")
        forbidden_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "REPLACE", "CREATE"]
        for kw in forbidden_keywords:
            if re.search(rf"\b{kw}\b", query):
                raise ValueError(f"Query contains forbidden keyword: {kw}")
        if len([s.strip() for s in self.query.split(";") if s.strip()]) > 1:
            raise ValueError("Multiple statements detected; only single SELECT allowed.")
        return True

    def execute_query(self):
        with sqlite3.connect(self.db_name) as conn:
            self.df = pd.read_sql_query(self.query, conn)
        return self.df

    def run(self) -> bool:
        self.previous_queries = []
        self.errors = []
        self.attempt = 0
        self.tokens = 0
        self.df = pd.DataFrame()

        while self.attempt < self.max_retries:
            if self.attempt == 0:
                self.generate_query()
            else:
                self.refine_query()

            try:
                self.check_query()
                self.execute_query()
            except Exception as e:
                self.errors.append(str(e))
                continue

            if self.df.empty:
                self.errors.append("Query returned empty result.")
                continue

            self.failed = False
            return self.failed

        # Retries exhausted
        self.failed = True
        return self.failed
    
    def wrap(self):
        if self.failed:
            answer = f"I couldn't generate a valid query for your request. Here are the issues I ran into: {', '.join(self.errors)}. Could you rephrase or provide more details?"
        elif not self.failed and self.df.shape[0] == 1:
            table = self.df.to_string(index=False)
            prompt = read_prompt(
                "prompts\sql_agent\describe_query.txt", 
                user_input=self.user_input,
                query=self.query, 
                table=table
            )
            response = self.model.invoke(prompt)
            answer = response.content
            self.tokens += response.response_metadata['token_usage']['total_tokens']
        else:
            answer = "The result is quite large, so I've displayed the data instead. You can ask me to summarize it or narrow down your request."
            display(self.df)
        return answer

    def chat(self, user_input):
        self.user_input = user_input
        self.run()
        answer = self.wrap()
        return answer