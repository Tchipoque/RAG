from pydantic import BaseModel, Field
import uuid
from typing import List, Union


class MinimalSource(BaseModel):
	file_path: str
	first_character_index:  int
	last_character_index:  int

class UnanswerdQuestioin(BaseModel):
	question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
	question: str

class AnsweredQuestion(UnanswerdQuestioin):
	sources: List[MinimalSource]
	answer: str

class RagDataset(BaseModel):
	rag_questions: List[Union[AnsweredQuestion, UnanswerdQuestioin]]

class MinimalSearchResults(BaseModel):
	question_id: str
	question: str
	retrieved_sources: List[MinimalSource]

class MinimalAnswer(MinimalSearchResults):
	answer: str

class StudentSearchResults(BaseModel):
	search_results: List[MinimalSearchResults]
	k: int

class StudentSearchResultsAndAnswer(BaseModel):
	search_results: List[MinimalAnswer]
	k: int


