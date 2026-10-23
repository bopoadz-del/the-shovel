"""The development API key for the test suite.

``CEREBRUM_DEV_KEY`` when the run sets one (CI sets a random value per run),
else a random value for this run. The app reads the same variable
(app.core.auth.dev_key), so no key is ever written in the source.
"""
import os
import secrets

DEV_KEY = os.environ.setdefault("CEREBRUM_DEV_KEY", "dev-" + secrets.token_hex(16))
