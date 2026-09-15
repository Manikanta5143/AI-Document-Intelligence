"""Database setup utility script.

Creates the MySQL database (if it doesn't already exist) and initializes
all tables defined in SQLAlchemy models.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path when executed directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pymysql
from app.config import get_settings
from app.database.session import init_db
from app.utils.logger import setup_logger

logger = setup_logger("db_setup")


def create_database_if_not_exists() -> bool:
    """Connect to MySQL server and ensure the target database exists."""
    settings = get_settings()
    logger.info("Connecting to MySQL server at %s:%s...", settings.mysql_host, settings.mysql_port)
    try:
        connection = pymysql.connect(
            host=settings.mysql_host,
            port=settings.mysql_port,
            user=settings.mysql_user,
            password=settings.mysql_password,
            charset="utf8mb4",
        )
        with connection.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{settings.mysql_database}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
            )
            logger.info("Database '%s' is verified/created.", settings.mysql_database)
        connection.close()
        return True
    except pymysql.MySQLError as err:
        logger.error(
            "Failed to connect to MySQL server: %s. "
            "Please check your credentials in .env (MYSQL_USER, MYSQL_PASSWORD).",
            err,
        )
        return False


def main():
    print("================================================================")
    print("AI Insurance Document Intelligence Assistant - DB Setup")
    print("================================================================")
    db_created = create_database_if_not_exists()
    if not db_created:
        print("\n[!] Setup incomplete: Unable to connect to MySQL server.")
        print("    1. Ensure the MySQL service is running.")
        print("    2. Ensure your password is set in .env:")
        print("       MYSQL_USER=root")
        print("       MYSQL_PASSWORD=your_password")
        print("       MYSQL_DATABASE=insurance_ai_db")
        sys.exit(1)

    tables_created = init_db()
    if tables_created:
        print("\n[✓] Database and tables successfully created and verified!")
    else:
        print("\n[!] Could not create tables. Check the error log above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
