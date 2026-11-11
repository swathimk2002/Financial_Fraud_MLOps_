
import os

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from dotenv import load_dotenv

load_dotenv("docker/postgres/.env")

def get_database_url() -> URL:
    """Build the PostgreSQL connection URL from environment variables."""

    required_variables = [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DB",
    ]

    missing = [
        variable
        for variable in required_variables
        if not os.getenv(variable)
    ]

    if missing:
        raise EnvironmentError(
            f"Missing database environment variables: {missing}"
        )

    return URL.create(
        drivername="postgresql+psycopg",
        username=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        database=os.environ["POSTGRES_DB"],
    )


def get_engine() -> Engine:
    """Create a reusable SQLAlchemy engine."""

    return create_engine(
        get_database_url(),
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )


def test_connection() -> bool:
    """Check whether PostgreSQL accepts a connection."""

    engine = get_engine()

    try:
        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT current_database(), current_user")
            )
            database, user = result.one()

            print("PostgreSQL connection successful!")
            print(f"Database: {database}")
            print(f"User: {user}")

        return True

    finally:
        engine.dispose()
