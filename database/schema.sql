-- Predictive Maintenance Database Schema
CREATE DATABASE IF NOT EXISTS qlbaotri;
USE qlbaotri;

-- Machines table with SMART sensor fields
CREATE TABLE IF NOT EXISTS machines (
    Machine_ID INT AUTO_INCREMENT PRIMARY KEY,
    Date DATE,
    Operating_Hours FLOAT DEFAULT 0,
    Temperature FLOAT,
    Vibration FLOAT,
    Pressure FLOAT,
    Production_Output FLOAT DEFAULT 0,
    Humidity FLOAT DEFAULT 50.0,
    Energy_consumption FLOAT DEFAULT 2.0,
    Machine_status INT DEFAULT 1,
    Anomaly_flag INT DEFAULT 0,
    Failure_type VARCHAR(50) DEFAULT 'Normal',
    Downtime_risk FLOAT DEFAULT 0.0,
    Maintenance_required INT DEFAULT 0,
    Maintenance_History TEXT,
    Failure INT DEFAULT 0
);

-- Maintenance records table
CREATE TABLE IF NOT EXISTS maintenance (
    Maintenance_ID INT AUTO_INCREMENT PRIMARY KEY,
    Machine_ID INT,
    Maintenance_Date DATE,
    Deadline DATE,
    Status VARCHAR(20) DEFAULT 'Scheduled',
    Technician VARCHAR(255),
    Cost DECIMAL(10,2),
    Description TEXT,
    FOREIGN KEY (Machine_ID) REFERENCES machines(Machine_ID)
);
