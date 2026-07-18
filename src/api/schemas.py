from pydantic import BaseModel

class ProjectBrief(BaseModel):
    description: str