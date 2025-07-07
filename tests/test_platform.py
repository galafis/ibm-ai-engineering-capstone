#!/usr/bin/env python3
"""
Unit tests for the platform
"""

import unittest
import sys
import os
import sqlite3
import tempfile

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestPlatform(unittest.TestCase):
    """Test cases for the platform"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_db = tempfile.mktemp(suffix='.db')
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
    
    def test_database_connection(self):
        """Test database connection"""
        conn = sqlite3.connect(self.test_db)
        self.assertIsNotNone(conn)
        conn.close()
    
    def test_data_insertion(self):
        """Test data insertion"""
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE test_data (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value REAL
            )
        """)
        
        cursor.execute("INSERT INTO test_data (name, value) VALUES (?, ?)", ("Test", 100.0))
        conn.commit()
        
        cursor.execute("SELECT COUNT(*) FROM test_data")
        count = cursor.fetchone()[0]
        
        self.assertEqual(count, 1)
        conn.close()
    
    def test_data_retrieval(self):
        """Test data retrieval"""
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE test_data (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value REAL
            )
        """)
        
        cursor.execute("INSERT INTO test_data (name, value) VALUES (?, ?)", ("Test", 100.0))
        conn.commit()
        
        cursor.execute("SELECT name, value FROM test_data WHERE id = 1")
        result = cursor.fetchone()
        
        self.assertEqual(result[0], "Test")
        self.assertEqual(result[1], 100.0)
        conn.close()

if __name__ == '__main__':
    unittest.main()
