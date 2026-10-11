"""Create the Superset account admin, or set its password to SUPERSET_ADMIN_PASSWORD if it exists.

Runs from start.sh before the server starts. The password is read from the environment, so it never
appears in a command line the way `superset fab create-admin --password` would put it.
"""

import os

from superset.app import create_app

USERNAME = "admin"

app = create_app()
with app.app_context():
    from superset import security_manager

    password = os.environ["SUPERSET_ADMIN_PASSWORD"]
    user = security_manager.find_user(username=USERNAME)
    if user is None:
        role = security_manager.find_role("Admin")
        if not security_manager.add_user(USERNAME, "Radar", "Admin", "admin@radar.local", role, password=password):
            raise SystemExit(f"could not create the account {USERNAME}")
        print(f"created the account {USERNAME}")
    else:
        security_manager.reset_password(user.id, password)
        print(f"set the password of {USERNAME}")
