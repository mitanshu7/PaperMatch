from pydantic import BaseModel, HttpUrl, field_validator, Field
import re
from datetime import datetime


# Create a request model
class TextRequest(BaseModel):
    text: str


arxiv_url_regex = re.compile(r".+arxiv\.org.+")
current_year = datetime.now().year

# Response model for arxiv papers
class ArxivPaper(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    authors: list[str] = Field(min_length=1)
    abstract: str = Field(min_length=1)
    url: HttpUrl = Field(min_length=1)
    pdf: HttpUrl = Field(min_length=1)
    month: int = Field(ge=1, le=12)
    year: int = Field(ge=1991, le=current_year)
    categories: list[str] = Field(min_length=1)
    
    @field_validator("url", "pdf")
    @classmethod
    def check_arxiv_url(cls, value: HttpUrl) -> HttpUrl:
        
        match = arxiv_url_regex.findall(str(value))
        
        if len(match) == 0:
            raise ValueError(f"{value} is not an arxiv url")
            
        return value
