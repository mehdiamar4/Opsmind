import os
import sys
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool


config = context.config

backend_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_dir))

database_url = os.environ.get("DATABASE_URL")
if not database_url:
    raise RuntimeError("DATABASE_URL must be set before running Alembic")
config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))

from app.database import Base  # noqa: E402
from app import models  # noqa: E402,F401


target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
