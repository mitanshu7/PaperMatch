from pydantic import BaseModel, HttpUrl, field_validator
import re

# Create a request model
class TextRequest(BaseModel):
    text: str


arxiv_url_regex = re.compile(r".+arxiv\.org.+")

# Response model for arxiv papers
class ArxivPaper(BaseModel):
    id: str 
    title: str
    authors: list[str]
    abstract: str
    url: HttpUrl
    pdf: HttpUrl
    month: int
    year: int
    categories: list[str]
    
    @field_validator("url", "pdf")
    @classmethod
    def check_arxiv_url(cls, value: HttpUrl) -> HttpUrl:
        
        match = arxiv_url_regex.findall(str(value))
        
        if len(match) == 0:
            raise ValueError(f"{value} is not an arxiv url")
            
        return value