import sqlite3
import re
import pandas as pd
from IPython.display import display
from utils import read_prompt


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

    def generate_query(self, refine=False):
        if refine:
            prompt = read_prompt(
                "prompts/sql_agent/refine_query.txt", 
                db_description=self.db_description, 
                user_input=self.user_input,
                previous_queries=", ".join(self.previous_queries),
                errors=", ".join(self.errors)
            )
        else:
            prompt = read_prompt(
                "prompts/sql_agent/generate_query.txt", 
                db_description=self.db_description,
                user_input=self.user_input
        )
        
        
        response = self.model.invoke(prompt)
        self.query = response.text
        self.previous_queries.append(self.query)
        self.attempt += 1
        self.tokens += response.usage_metadata['total_tokens']
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
                self.generate_query(refine=True)

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
            answer = response.text
            self.tokens += response.usage_metadata['total_tokens']
        else:
            answer = "The result is quite large, so I've displayed the data instead. You can ask me to summarize it or narrow down your request."
            display(self.df)
        return answer

    def chat(self, user_input):
        self.user_input = user_input
        self.run()
        answer = self.wrap()
        return answer