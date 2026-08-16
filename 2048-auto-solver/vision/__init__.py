"""Perception: turning screen pixels into a board the solver understands.

Everything here consumes NumPy BGR frames produced by a ``backends.capture`` backend and
never talks to the OS directly -- that boundary is what lets ``core/`` stay pure and lets this
package be exercised in tests with plain image fixtures instead of a live screen.
"""
