"""Define the state structures for the agent."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Annotated, Sequence, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


@dataclass
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    sender: str
