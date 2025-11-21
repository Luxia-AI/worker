#!/usr/bin/env python3
"""
Load testing for Luxia Worker logging system.
Tests concurrent requests and system performance under load.
"""
import asyncio
import sys
import time
from datetime import datetime
from typing import Dict, List

try:
    import aiohttp
except ImportError:
    print("Error: aiohttp package not installed")
    print("Install with: pip install aiohttp")
    sys.exit(1)


class LoadTester:
    """Performance load tester for API endpoints."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.results: List[Dict] = []

    async def single_request(self, session: aiohttp.ClientSession, endpoint: str, request_id: int):
        """Make a single API request and record metrics."""
        start = time.time()

        try:
            async with session.get(f"{self.base_url}{endpoint}") as resp:
                duration = time.time() - start
                body = await resp.text()

                result = {
                    "request_id": request_id,
                    "endpoint": endpoint,
                    "status": resp.status,
                    "duration": duration,
                    "success": resp.status == 200,
                    "body_size": len(body),
                    "timestamp": datetime.now().isoformat(),
                }

                self.results.append(result)
                return result

        except Exception as e:
            duration = time.time() - start
            result = {
                "request_id": request_id,
                "endpoint": endpoint,
                "status": 0,
                "duration": duration,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            self.results.append(result)
            return result

    async def concurrent_load_test(self, endpoint: str, concurrent: int, total_requests: int):
        """
        Run concurrent load test.

        Args:
            endpoint: API endpoint to test
            concurrent: Number of concurrent requests
            total_requests: Total number of requests to make
        """
        print(f"\n🔥 Load Test: {endpoint}")
        print(f"   Concurrent: {concurrent} | Total: {total_requests}")
        print("-" * 70)

        connector = aiohttp.TCPConnector(limit=concurrent)
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []

            for i in range(total_requests):
                task = self.single_request(session, endpoint, i + 1)
                tasks.append(task)

                # Add small delay every 'concurrent' requests to control rate
                if (i + 1) % concurrent == 0:
                    await asyncio.sleep(0.1)

            # Wait for all requests to complete
            print(f"⏳ Executing {total_requests} requests...")
            start_time = time.time()

            await asyncio.gather(*tasks, return_exceptions=True)

            total_duration = time.time() - start_time

            # Calculate statistics
            successful = sum(1 for r in self.results if isinstance(r, dict) and r.get("success"))
            failed = total_requests - successful

            durations = [r["duration"] for r in self.results if isinstance(r, dict) and "duration" in r]
            avg_duration = sum(durations) / len(durations) if durations else 0
            min_duration = min(durations) if durations else 0
            max_duration = max(durations) if durations else 0

            requests_per_sec = total_requests / total_duration if total_duration > 0 else 0

            # Print results
            print("\n✅ Load Test Complete")
            print(f"   Total Duration: {total_duration:.2f}s")
            print(f"   Successful: {successful}/{total_requests} ({successful/total_requests*100:.1f}%)")
            print(f"   Failed: {failed}")
            print(f"   Requests/sec: {requests_per_sec:.2f}")
            print(f"   Avg Response: {avg_duration*1000:.2f}ms")
            print(f"   Min Response: {min_duration*1000:.2f}ms")
            print(f"   Max Response: {max_duration*1000:.2f}ms")

            return {
                "endpoint": endpoint,
                "total_requests": total_requests,
                "successful": successful,
                "failed": failed,
                "total_duration": total_duration,
                "requests_per_sec": requests_per_sec,
                "avg_duration_ms": avg_duration * 1000,
                "min_duration_ms": min_duration * 1000,
                "max_duration_ms": max_duration * 1000,
            }

    async def stress_test(self):
        """Run comprehensive stress test suite."""
        print("=" * 70)
        print("Load & Performance Test - Luxia Worker")
        print("=" * 70)

        # Test 1: Light load - logs endpoint
        self.results = []
        await self.concurrent_load_test("/admin/logs?limit=5", concurrent=10, total_requests=50)

        # Test 2: Medium load - logs endpoint
        self.results = []
        await self.concurrent_load_test("/admin/logs?limit=10", concurrent=20, total_requests=100)

        # Test 3: Heavy load - logs endpoint
        self.results = []
        await self.concurrent_load_test("/admin/logs?limit=1", concurrent=50, total_requests=200)

        # Test 4: Stats endpoint
        self.results = []
        await self.concurrent_load_test("/admin/logs/stats", concurrent=10, total_requests=30)

        print("\n" + "=" * 70)
        print("✅ All load tests completed")
        print("=" * 70)


async def check_service_health(base_url: str) -> bool:
    """Check if service is healthy before running tests."""
    print(f"🔍 Checking service health at {base_url}...")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/admin/logs?limit=1", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    print(f"✅ Service is healthy (status: {resp.status})\n")
                    return True
                else:
                    print(f"⚠️  Service returned status: {resp.status}\n")
                    return False
    except Exception as e:
        print(f"❌ Service unreachable: {e}\n")
        return False


async def monitor_container_stats():
    """Monitor Docker container resource usage."""
    print("\n📊 Container Resource Usage:")
    print("-" * 70)

    try:
        # Get container stats
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "stats",
            "luxia-worker",
            "--no-stream",
            "--format",
            "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode == 0:
            print(stdout.decode())
        else:
            print(f"⚠️  Could not get stats: {stderr.decode()}")

    except Exception as e:
        print(f"⚠️  Error getting container stats: {e}")


async def main():
    """Main test orchestrator."""
    base_url = "http://localhost:9000"

    # Check service health first
    if not await check_service_health(base_url):
        print("❌ Service is not healthy. Please ensure Docker containers are running.")
        print("   Run: docker-compose ps")
        sys.exit(1)

    # Get initial container stats
    await monitor_container_stats()

    # Run load tests
    tester = LoadTester(base_url)
    await tester.stress_test()

    # Get final container stats
    await monitor_container_stats()

    print("\n💡 Tip: View detailed logs with:")
    print("   docker logs luxia-worker --tail 50")
    print("   curl http://localhost:9000/admin/logs/stats | python -m json.tool")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(0)
