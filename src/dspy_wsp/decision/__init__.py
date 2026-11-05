"""System One decision models: request schema, model networks and serving.

A decision model reads a state and typed questions (Choice, Noul, Score) and
returns a probability for every option of every question, with no text
generation. Requests and answers follow TypeSafe's System One API.
"""
