#!/usr/bin/env python3
"""
WebSocket streaming test for Luxia Worker logging system.
Tests realtime log streaming via WebSocket connection.
"""
import asyncio
import json
import sys
from datetime import datetime

try:
    import websockets
except ImportError:
    print("Error: websockets package not installed")
    print("Install with: pip install websockets")
    sys.exit(1)


async def test_websocket_streaming(url: str, duration: int = 10):
    """
    Connect to WebSocket and listen for log messages.

    Args:
        url: WebSocket URL (e.g., ws://localhost:9000/admin/logs/stream)
        duration: How long to listen (seconds)
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to {url}")

    try:
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Connected!")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Listening for {duration} seconds...\n")

            message_count = 0
            start_time = asyncio.get_event_loop().time()

            try:
                while True:
                    # Check if duration expired
                    if asyncio.get_event_loop().time() - start_time > duration:
                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Duration expired, closing...")
                        break

                    # Wait for message with timeout
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        message_count += 1

                        # Parse and display log
                        try:
                            log = json.loads(message)
                            timestamp = log.get("timestamp", "N/A")
                            level = log.get("level", "INFO")
                            module = log.get("module", "unknown")
                            msg = log.get("message", "")

                            print(f"[{message_count:3d}] {timestamp} [{level:5s}] {module}")
                            print(f"      {msg}")
                            print()

                        except json.JSONDecodeError:
                            print(f"[{message_count:3d}] Raw: {message}")

                    except asyncio.TimeoutError:
                        # No message received, continue
                        pass

            except KeyboardInterrupt:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Interrupted by user")

            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Summary:")
            print(f"  Messages received: {message_count}")
            print(f"  Duration: {duration}s")

    except websockets.exceptions.WebSocketException as e:
        print(f"❌ WebSocket error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    return True


async def generate_test_logs(base_url: str, count: int = 5):
    """
    Generate test logs by making API calls.

    Args:
        base_url: Base HTTP URL (e.g., http://localhost:9000)
        count: Number of test requests to make
    """
    import aiohttp

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Generating {count} test logs...")

    async with aiohttp.ClientSession() as session:
        for i in range(count):
            try:
                async with session.get(f"{base_url}/admin/logs?limit=1") as resp:
                    if resp.status == 200:
                        print(f"  [{i+1}/{count}] ✅ Generated log via API call")
                    else:
                        print(f"  [{i+1}/{count}] ⚠️  API returned {resp.status}")
                await asyncio.sleep(0.5)  # Small delay between requests
            except Exception as e:
                print(f"  [{i+1}/{count}] ❌ Error: {e}")


async def main():
    """Main test orchestrator."""
    ws_url = "ws://localhost:9000/admin/logs/stream?channel=logs:all"
    http_url = "http://localhost:9000"

    print("=" * 70)
    print("WebSocket Streaming Test - Luxia Worker")
    print("=" * 70)

    # Test 1: Connect and listen (background)
    print("\n📡 Test 1: WebSocket Connection & Streaming")
    print("-" * 70)

    # Start listening task
    listen_task = asyncio.create_task(test_websocket_streaming(ws_url, duration=15))

    # Wait for connection to establish
    await asyncio.sleep(2)

    # Generate some test traffic
    await generate_test_logs(http_url, count=10)

    # Wait for listening to complete
    success = await listen_task

    if success:
        print("\n✅ WebSocket streaming test PASSED")
    else:
        print("\n❌ WebSocket streaming test FAILED")

    print("=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)
