import os
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from firecrawl.firecrawl import FirecrawlApp
import requests

# Change to localhost:3002 if running locally
#firecrawl = FirecrawlApp(api_url="http://host.docker.internal:3002", api_key="123")

# Uncomment for cloud version
firecrawl = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))


serper_headers = {
    "X-API-KEY": os.getenv("SERPER_API_KEY"),
    "Content-Type": "application/json",
}


class LeadFinderInput(BaseModel):
    query: str = Field(description="search query to look up leads")
    location: str = Field(
        description="Location of the query. E.g 'Atlanta, United States', or only the country if no city is provided."
    )


class LeadFinderTool(BaseTool):
    name: str = "LeadFinderTool"
    description: str = (
        "Find leads from Google Places. Use the query input parameter to search for the niche, and the location for the specific location"
    )
    args_schema: Type[BaseModel] = LeadFinderInput

    def _run(self, query: str, location: str):
        request_body = {"q": query, "location": location}
        response = requests.request(
            "POST",
            url="https://google.serper.dev/places",
            headers=serper_headers,
            json=request_body,
        )
        return response.json()["places"]


class ExtractionSchema(BaseModel):
    email_address: str
    address: str
    title: str
    phone: str
    CEO: str
    company_mission: str


class LeadExtractorInput(BaseModel):
    url: str = Field(description="Url to extract data from")


class LeadExtractorTool(BaseTool):
    name: str = "LeadExtractor"
    description: str = "A tool for extracting lead information from a given url."
    args_schema: Type[BaseModel] = LeadExtractorInput

    def _run(self, url: str):
        data = firecrawl.scrape_url(
            url,
            {
                "formats": ["extract"],
                "extract": {"schema": ExtractionSchema.model_json_schema()},
            },
        )
        return data
