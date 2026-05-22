#!/usr/bin/env python3
"""Create or update a frontend user. Run from the server (where Postgres is reachable).

Usage:
    python scripts/create_user.py --username demo --password demo
    python scripts/create_user.py --username demo  # prompts for password
    python scripts/create_user.py --username demo --deactivate
"""

from __future__ import annotations

import argparse
import getpass
import sys

import bcrypt

from rag.db import get_connection


def upsert_user(username: str, password: str | None, is_active: bool) -> None:
    with get_connection() as conn:
        row = conn.execute("SELECT id FROM users WHERE username = %s", (username,)).fetchone()
        if row:
            user_id = row[0]
            if password is not None:
                pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode()
                conn.execute(
                    "UPDATE users SET password_hash = %s, is_active = %s WHERE id = %s",
                    (pw_hash, is_active, user_id),
                )
                print(f"updated password for user {username}")
            else:
                conn.execute(
                    "UPDATE users SET is_active = %s WHERE id = %s",
                    (is_active, user_id),
                )
                print(f"updated user {username} (active={is_active})")
        else:
            if password is None:
                raise SystemExit("password is required when creating a new user")
            pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode()
            conn.execute(
                "INSERT INTO users (username, password_hash, is_active) VALUES (%s, %s, %s)",
                (username, pw_hash, is_active),
            )
            print(f"created user {username}")
        conn.commit()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create or update a frontend user.")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", help="If omitted, will prompt interactively.")
    parser.add_argument("--deactivate", action="store_true", help="Disable login for this user.")
    args = parser.parse_args(argv)

    password = args.password
    if password is None and not args.deactivate:
        password = getpass.getpass("password: ")
        confirm = getpass.getpass("confirm:  ")
        if password != confirm:
            print("passwords do not match", file=sys.stderr)
            return 2
    upsert_user(args.username, password, is_active=not args.deactivate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
