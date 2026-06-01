# Agent Basics

An AI agent is a program that receives a goal, decides what action to take,
uses tools when needed, observes tool results, and returns a final answer.

The most important beginner pattern is the agent loop: think, act, observe,
and answer. The loop can be controlled by simple rules at first, then later by
an LLM that chooses tools through function calling.

Tools are ordinary functions with clear inputs and outputs. Common tools include
calculators, document search, database queries, web search, email APIs, file
operations, and business workflow APIs.
