import os
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from firecrawl.firecrawl import FirecrawlApp
import requests

headers = {"X-API-KEY": os.getenv("SERPER_API_KEY"), "Content-Type": "application/json"}

# Use below when running locally
# firecrawl = FirecrawlApp(api_url="http://localhost:3002", api_key='test')
firecrawl = FirecrawlApp(api_url="http://host.docker.internal:3002", api_key="test")


# class SiteMapperInput(BaseModel):
#     url: str = Field(description="Url to map")


# # Firecrawl Map tool
# class SiteMapperTool(BaseTool):
#     name: str = "SiteMapperTool"
#     description: str = (
#         "Use this tool to gather all the pages of a given url. Do not include a trailing slash when inputting a url"
#     )
#     args_schema: Type[BaseModel] = SiteMapperInput

#     def __run__(self, url: str):
#         # Check if the url has a trailing slash and remove it
#         if url[-1] == "/":
#             url = url[:-1]
#         return firecrawl.map_url(url=url)


# TODO: Use below module instead of calling serper API directly.
# search = GoogleSerperAPIWrapper(type="places", gl="us")


class ContactStructure(BaseModel):
    email_address: str
    address: str
    title: str
    phone: str
    CEO: str
    company_mission: str


class LeadFinderInput(BaseModel):
    query: str = Field(description="search query to look up leads")
    location: str = Field(
        description="Location of the query. E.g 'Atlanta, United States', or only the country if no city is provided."
    )


class GooglePlacesTool(BaseTool):
    name: str = "GooglePlacesTool"
    description: str = (
        "Find leads via Google Places. Use the query input parameter to search for the niche, and the location for the specific location"
    )
    args_schema: Type[BaseModel] = LeadFinderInput

    def _run(self, query: str, location: str):
        request_body = {"q": query, "location": location}
        response = requests.request(
            "POST",
            url="https://google.serper.dev/places",
            headers=headers,
            json=request_body,
        )
        return response.json()["places"]


class LeadExtractorInput(BaseModel):
    url: str = Field(description="Url to extract data from")


class LeadExtractorTool(BaseTool):
    """Extracts lead information from a web page."""

    name: str = "LeadExtractor"
    description: str = "A tool for extracting lead information from a given url."

    args_schema: Type[BaseModel] = LeadExtractorInput

    def _run(self, url: str):
        data = firecrawl.scrape_url(
            url,
            {
                "formats": ["extract"],
                "extract": {"schema": ContactStructure.model_json_schema()},
            },
        )
        return data
