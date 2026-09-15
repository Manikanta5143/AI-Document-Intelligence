-- ==============================================================================
-- AI Insurance Document Intelligence Assistant: Snowflake Setup DDL
-- Database, Warehouse, Schemas, Analytical Tables, and S3 Integration
-- ==============================================================================

-- 1. Setup Role and Virtual Warehouse
USE ROLE ACCOUNTADMIN;

CREATE WAREHOUSE IF NOT EXISTS COMPUTE_WH
    WITH WAREHOUSE_SIZE = 'X-SMALL'
    AUTO_SUSPEND = 300
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'Virtual warehouse for Insurance AI document analytics';

-- 2. Setup Database and Schema
CREATE DATABASE IF NOT EXISTS INSURANCE_DB
    COMMENT = 'Analytical database for insurance document intelligence';

USE DATABASE INSURANCE_DB;

CREATE SCHEMA IF NOT EXISTS PUBLIC
    COMMENT = 'Default analytical schema for insurance metrics and reporting';

USE SCHEMA PUBLIC;

-- 3. Core Analytical Tables
-- ------------------------------------------------------------------------------
-- A. Document Analytics (High-level document processing metrics)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_analytics (
    document_id INTEGER PRIMARY KEY,
    document_type VARCHAR(100),
    file_name VARCHAR(255),
    processing_time FLOAT,
    chunk_count INTEGER,
    query_count INTEGER DEFAULT 0,
    risk_category VARCHAR(50) DEFAULT 'LOW',
    status VARCHAR(50) DEFAULT 'PROCESSED',
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    synced_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ------------------------------------------------------------------------------
-- B. Policy Metrics (Structured insurance parameters extracted from documents)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS policy_metrics (
    policy_id VARCHAR(100) PRIMARY KEY,
    document_id INTEGER,
    policy_type VARCHAR(100),
    deductible NUMBER(12, 2),
    coverage_amount NUMBER(14, 2),
    premium NUMBER(12, 2),
    risk_level VARCHAR(50),
    extracted_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ------------------------------------------------------------------------------
-- C. Query Analytics (User question frequency, latency, and engagement)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS query_analytics (
    query_id VARCHAR(64) PRIMARY KEY,
    document_id INTEGER,
    question_text VARCHAR(2000),
    question_category VARCHAR(100),
    answer_length INTEGER,
    retrieved_chunks_count INTEGER,
    queried_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ==============================================================================
-- 4. S3 ↔ Snowflake Storage Integration & External Stage (Optional Advanced Step)
-- ==============================================================================
-- To query raw documents or metadata files directly from S3 without ingestion:
--
-- Step A: Create an AWS IAM Role with trust policy allowing Snowflake.
-- Step B: Create a Storage Integration in Snowflake:
--
-- CREATE OR REPLACE STORAGE INTEGRATION s3_insurance_integration
--   TYPE = EXTERNAL_STAGE
--   STORAGE_PROVIDER = 'S3'
--   ENABLED = TRUE
--   STORAGE_AWS_ROLE_ARN = 'arn:aws:iam::260172549174:role/snowflake-s3-access-role'
--   STORAGE_ALLOWED_LOCATIONS = ('s3://verisk-bucket-12223/documents/');
--
-- Step C: Retrieve the Snowflake IAM User ARN to update AWS Trust Policy:
-- DESC STORAGE INTEGRATION s3_insurance_integration;
-- (Look for STORAGE_AWS_IAM_USER_ARN and STORAGE_AWS_EXTERNAL_ID)
--
-- Step D: Create the External Stage:
-- CREATE OR REPLACE STAGE insurance_s3_stage
--   STORAGE_INTEGRATION = s3_insurance_integration
--   URL = 's3://verisk-bucket-12223/documents/'
--   FILE_FORMAT = (TYPE = 'JSON');
-- ==============================================================================
