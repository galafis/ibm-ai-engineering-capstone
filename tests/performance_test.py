#!/usr/bin/env python3
"""
Performance Tests for AI Engineering Platform
Ibm Ai Engineering Capstone
"""

import time
import requests
import threading
import statistics
import sqlite3
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceTest:
    """Performance testing suite for the platform"""
    
    def __init__(self):
        self.base_url = "http://localhost:8501"
        self.db_path = "platform.db"
        self.results = {}
    
    def test_database_performance(self):
        """Test database operations performance"""
        logger.info("Testing database performance...")
        
        # Test connection time
        start_time = time.time()
        conn = sqlite3.connect(self.db_path)
        connection_time = time.time() - start_time
        
        # Test query performance
        start_time = time.time()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM platform_data")
        result = cursor.fetchone()
        query_time = time.time() - start_time
        
        # Test insert performance
        start_time = time.time()
        cursor.execute("""
            INSERT INTO platform_data (name, category, value, status)
            VALUES (?, ?, ?, ?)
        """, ("Test Item", "Test Category", 100.0, "active"))
        conn.commit()
        insert_time = time.time() - start_time
        
        conn.close()
        
        self.results['database'] = {
            'connection_time': connection_time,
            'query_time': query_time,
            'insert_time': insert_time,
            'record_count': result[0] if result else 0
        }
        
        logger.info(f"Database performance: Connection={connection_time:.4f}s, Query={query_time:.4f}s, Insert={insert_time:.4f}s")
    
    def test_memory_usage(self):
        """Test memory usage of the application"""
        logger.info("Testing memory usage...")
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        self.results['memory'] = {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            'percent': process.memory_percent()
        }
        
        logger.info(f"Memory usage: RSS={memory_info.rss / 1024 / 1024:.2f}MB, VMS={memory_info.vms / 1024 / 1024:.2f}MB")
    
    def test_cpu_usage(self):
        """Test CPU usage during operations"""
        logger.info("Testing CPU usage...")
        
        # Monitor CPU for 5 seconds
        cpu_percentages = []
        for _ in range(5):
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_percentages.append(cpu_percent)
        
        self.results['cpu'] = {
            'average_percent': statistics.mean(cpu_percentages),
            'max_percent': max(cpu_percentages),
            'min_percent': min(cpu_percentages)
        }
        
        logger.info(f"CPU usage: Avg={statistics.mean(cpu_percentages):.2f}%, Max={max(cpu_percentages):.2f}%")
    
    def test_data_processing_speed(self):
        """Test data processing speed"""
        logger.info("Testing data processing speed...")
        
        # Test processing 1000 records
        start_time = time.time()
        
        # Simulate data processing
        data = []
        for i in range(1000):
            record = {
                'id': i,
                'name': f'Item {i}',
                'value': i * 1.5,
                'processed': True
            }
            data.append(record)
        
        # Simulate calculations
        total_value = sum(record['value'] for record in data)
        average_value = total_value / len(data)
        
        processing_time = time.time() - start_time
        
        self.results['data_processing'] = {
            'records_processed': len(data),
            'processing_time': processing_time,
            'records_per_second': len(data) / processing_time,
            'total_value': total_value,
            'average_value': average_value
        }
        
        logger.info(f"Data processing: {len(data)} records in {processing_time:.4f}s ({len(data) / processing_time:.2f} records/sec)")
    
    def test_concurrent_operations(self, num_threads=10):
        """Test concurrent operations performance"""
        logger.info(f"Testing concurrent operations with {num_threads} threads...")
        
        def worker_task(thread_id):
            """Worker task for concurrent testing"""
            start_time = time.time()
            
            # Simulate database operation
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM platform_data")
            result = cursor.fetchone()
            conn.close()
            
            # Simulate processing
            time.sleep(0.1)  # Simulate work
            
            end_time = time.time()
            return {
                'thread_id': thread_id,
                'execution_time': end_time - start_time,
                'result': result[0] if result else 0
            }
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_threads)]
            results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        execution_times = [r['execution_time'] for r in results]
        
        self.results['concurrent'] = {
            'num_threads': num_threads,
            'total_time': total_time,
            'average_execution_time': statistics.mean(execution_times),
            'max_execution_time': max(execution_times),
            'min_execution_time': min(execution_times),
            'throughput': num_threads / total_time
        }
        
        logger.info(f"Concurrent operations: {num_threads} threads in {total_time:.4f}s ({num_threads / total_time:.2f} ops/sec)")
    
    def test_load_simulation(self, duration_seconds=30):
        """Simulate load for a specified duration"""
        logger.info(f"Running load simulation for {duration_seconds} seconds...")
        
        start_time = time.time()
        operations_completed = 0
        
        while time.time() - start_time < duration_seconds:
            # Simulate various operations
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Random operations
            import random
            operation = random.choice(['select', 'insert', 'update'])
            
            if operation == 'select':
                cursor.execute("SELECT * FROM platform_data LIMIT 10")
                cursor.fetchall()
            elif operation == 'insert':
                cursor.execute("""
                    INSERT INTO platform_data (name, category, value, status)
                    VALUES (?, ?, ?, ?)
                """, (f"Load Test {operations_completed}", "Load Test", random.uniform(1, 100), "active"))
                conn.commit()
            elif operation == 'update':
                cursor.execute("""
                    UPDATE platform_data 
                    SET value = ? 
                    WHERE id = (SELECT id FROM platform_data ORDER BY RANDOM() LIMIT 1)
                """, (random.uniform(1, 100),))
                conn.commit()
            
            conn.close()
            operations_completed += 1
            
            # Small delay to prevent overwhelming
            time.sleep(0.01)
        
        total_time = time.time() - start_time
        
        self.results['load_simulation'] = {
            'duration': total_time,
            'operations_completed': operations_completed,
            'operations_per_second': operations_completed / total_time
        }
        
        logger.info(f"Load simulation: {operations_completed} operations in {total_time:.2f}s ({operations_completed / total_time:.2f} ops/sec)")
    
    def run_all_tests(self):
        """Run all performance tests"""
        logger.info("Starting comprehensive performance test suite...")
        
        tests = [
            self.test_database_performance,
            self.test_memory_usage,
            self.test_cpu_usage,
            self.test_data_processing_speed,
            lambda: self.test_concurrent_operations(10),
            lambda: self.test_load_simulation(30)
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                logger.error(f"Test failed: {e}")
        
        self.generate_report()
    
    def generate_report(self):
        """Generate performance test report"""
        logger.info("Generating performance test report...")
        
        print("\n" + "="*60)
        print("PERFORMANCE TEST REPORT")
        print("="*60)
        
        if 'database' in self.results:
            db = self.results['database']
            print(f"\n📊 DATABASE PERFORMANCE:")
            print(f"   Connection Time: {db['connection_time']:.4f}s")
            print(f"   Query Time: {db['query_time']:.4f}s")
            print(f"   Insert Time: {db['insert_time']:.4f}s")
            print(f"   Record Count: {db['record_count']:,}")
        
        if 'memory' in self.results:
            mem = self.results['memory']
            print(f"\n💾 MEMORY USAGE:")
            print(f"   RSS: {mem['rss_mb']:.2f} MB")
            print(f"   VMS: {mem['vms_mb']:.2f} MB")
            print(f"   Percentage: {mem['percent']:.2f}%")
        
        if 'cpu' in self.results:
            cpu = self.results['cpu']
            print(f"\n⚡ CPU USAGE:")
            print(f"   Average: {cpu['average_percent']:.2f}%")
            print(f"   Maximum: {cpu['max_percent']:.2f}%")
            print(f"   Minimum: {cpu['min_percent']:.2f}%")
        
        if 'data_processing' in self.results:
            dp = self.results['data_processing']
            print(f"\n🔄 DATA PROCESSING:")
            print(f"   Records Processed: {dp['records_processed']:,}")
            print(f"   Processing Time: {dp['processing_time']:.4f}s")
            print(f"   Records/Second: {dp['records_per_second']:.2f}")
        
        if 'concurrent' in self.results:
            conc = self.results['concurrent']
            print(f"\n🔀 CONCURRENT OPERATIONS:")
            print(f"   Threads: {conc['num_threads']}")
            print(f"   Total Time: {conc['total_time']:.4f}s")
            print(f"   Avg Execution: {conc['average_execution_time']:.4f}s")
            print(f"   Throughput: {conc['throughput']:.2f} ops/sec")
        
        if 'load_simulation' in self.results:
            load = self.results['load_simulation']
            print(f"\n🏋️ LOAD SIMULATION:")
            print(f"   Duration: {load['duration']:.2f}s")
            print(f"   Operations: {load['operations_completed']:,}")
            print(f"   Ops/Second: {load['operations_per_second']:.2f}")
        
        print("\n" + "="*60)
        print("✅ PERFORMANCE TEST COMPLETED")
        print("="*60)

def main():
    """Main function to run performance tests"""
    print("🚀 Starting AI Engineering Platform Performance Tests")
    
    # Initialize and run tests
    perf_test = PerformanceTest()
    perf_test.run_all_tests()

if __name__ == "__main__":
    main()
