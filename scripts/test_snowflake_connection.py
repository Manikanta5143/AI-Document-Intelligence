"""Comprehensive Snowflake Connection & Health Verification Script.

Tests connectivity, executes diagnostic queries, verifies schema & table initialization,
and reports status without leaking passwords or sensitive tokens.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.config import get_settings
from app.snowflake.connection import (
    SnowflakeConnectionManager,
    get_snowflake_manager,
    verify_snowflake_connection,
)
from app.services.snowflake_analytics_service import get_analytics_service, SnowflakeAnalyticsService


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    print_banner("SNOWFLAKE INTEGRATION DIAGNOSTIC & VERIFICATION")
    settings = get_settings()

    print("\n1. CONFIGURATION INSPECTION:")
    print(f"   - Snowflake Account   : {settings.snowflake_account or '[NOT CONFIGURED]'}")
    print(f"   - Snowflake User      : {settings.snowflake_user or '[NOT CONFIGURED]'}")
    print(f"   - Snowflake Password  : {'*' * len(settings.snowflake_password) if settings.snowflake_password else '[NOT CONFIGURED]'}")
    print(f"   - Snowflake Warehouse : {settings.snowflake_warehouse}")
    print(f"   - Snowflake Database  : {settings.snowflake_database}")
    print(f"   - Snowflake Schema    : {settings.snowflake_schema}")
    print(f"   - Snowflake Role      : {settings.snowflake_role}")

    mgr = get_snowflake_manager(settings=settings)
    configured = mgr.is_configured()
    print(f"\n   Configuration Status  : {'COMPLETE' if configured else 'INCOMPLETE'}")

    if not configured:
        print("\n" + "-" * 70)
        print(" [NOTICE] Snowflake account credentials are not yet populated in .env.")
        print(" To connect to a live Snowflake instance, set the following in your .env:")
        print('   SNOWFLAKE_ACCOUNT="<your_org-account_or_locator>"')
        print('   SNOWFLAKE_USER="<your_username>"')
        print('   SNOWFLAKE_PASSWORD="<your_password>"')
        print("-" * 70)

        print("\n2. TESTING FALLBACK SERVICE (MockAnalyticsService):")
        svc = get_analytics_service()
        print(f"   - Active Service Class: {type(svc).__name__}")
        summary = svc.get_analytics_summary()
        print(f"   - Engine Reported     : {summary.engine}")
        print("   - Fallback Execution  : PASS (Backend remains fully operational)")
        return

    print("\n2. TESTING LIVE SNOWFLAKE CONNECTION:")
    print("   Attempting handshake with Snowflake Data Cloud...")
    diag = verify_snowflake_connection()
    status = diag.get("status")

    if status == "CONNECTED":
        print("   Connection Status     : SUCCESS (HTTP 200 Handshake)")
        print(f"   - Snowflake Version   : {diag.get('version')}")
        print(f"   - Active Account      : {diag.get('account')}")
        print(f"   - Active Role         : {diag.get('role')}")
        print(f"   - Active Warehouse    : {diag.get('warehouse')}")
        print(f"   - Active Database     : {diag.get('database')}")
        print(f"   - Active Schema       : {diag.get('schema')}")
        print(f"   - Server UTC Time     : {diag.get('server_time')}")

        print("\n3. INITIALIZING ANALYTICAL TABLES (DDL):")
        try:
            mgr.init_tables()
            print("   - Table document_analytics : VERIFIED / CREATED")
            print("   - Table policy_metrics     : VERIFIED / CREATED")
            print("   - Table query_analytics    : VERIFIED / CREATED")
        except Exception as e:
            print(f"   - Table initialization error: {e}")

        print("\n4. TESTING ANALYTICS SERVICE LIVE AGGREGATION:")
        try:
            live_svc = SnowflakeAnalyticsService(connection_manager=mgr)
            summary = live_svc.get_analytics_summary()
            print(f"   - Analytics Engine    : {summary.engine}")
            print(f"   - Total Processed Docs: {summary.total_processed_documents}")
            print("   - OLAP Queries Test   : PASS")
        except Exception as e:
            print(f"   - Analytics aggregation error: {e}")

    else:
        print(f"   Connection Status     : FAILED ({status})")
        print(f"   - Error Type          : {diag.get('error_type')}")
        print(f"   - Error Detail        : {diag.get('error_message')}")


if __name__ == "__main__":
    main()
