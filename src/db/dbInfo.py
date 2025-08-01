#%%
from peewee import (
    SqliteDatabase,
    Model,
    CharField,
    DateTimeField,
    TextField
)

from datetime import datetime
import os
import json

db = SqliteDatabase(os.sep.join([os.path.dirname(__file__), "llm_data.db"]))

class JSONField(TextField):
    def db_value(self, value):
        return json.dumps(value)

    def python_value(self, value):
        if value is not None:
            return json.loads(value)

class BaseModel(Model):
    class Meta:
        database = db

class LLMAgentData(BaseModel):
    task_id = CharField(index=True)
    function_name = CharField()
    variable_name = CharField(null=True)
    data = JSONField(null=True) # Use our custom JSON field
    more_info = JSONField(null=True)
    created_on = DateTimeField(default=datetime.now)
    modified_on = DateTimeField(default=datetime.now)

def peewee_table():
    db.connect()
    db.create_tables([LLMAgentData], safe=True)
    db.close()

def add_and_query_peewee_data():
    db.connect()
    new_data_entry = LLMAgentData.create(
        task_id="peewee-task-002",
        function_name="extract_entities",
        type="output",
        variable_name="entities",
        data={"entities": ["Python", "SQLite", "Peewee"]}
    )
    print(f"Added new data entry: {new_data_entry}")

    all_entries = LLMAgentData.select()
    for entry in all_entries:
        print(f"ID: {entry.id}, TaskID: {entry.task_id}, Func: {entry.function_name}, Data: {entry.data}")
    db.close()
