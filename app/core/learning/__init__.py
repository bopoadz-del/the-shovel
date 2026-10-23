"""Learning subsystem — code that makes the platform smarter over time.

The :class:`app.blocks.learning_engine.LearningEngineBlock` is the public
surface ('hydrate', 'record_correction', 'tune', 'promote', 'record_pattern',
...). The heavy implementations live here so the block stays a thin
dispatcher and the logic is reusable by other call sites (the nightly
scheduler, future REPL tools, etc.).
"""
