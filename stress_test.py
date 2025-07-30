import asyncio
import aiohttp
import time
import json
from datetime import datetime

# Configuration
CHATBOT_URL = "http://localhost:8080"  # Adjust to your actual endpoint
LOGIN_ENDPOINT = f"{CHATBOT_URL}/api/auth/login"
CHAT_ENDPOINT = f"{CHATBOT_URL}/api/chat/stream"  # Using the streaming endpoint

# Test questions for sensation & perception
QUESTIONS = [
    "Explain the trichromatic theory and opponent-process theory of color vision.",
    "Describe the complete pathway of auditory processing from sound waves to brain perception.",
    "Compare bottom-up and top-down processing in visual perception with examples.",
    "What are the neural mechanisms underlying depth perception including binocular and monocular cues?",
    "Explain how attention affects sensory processing in the visual and auditory systems.",
    "Describe the difference between sensation and perception with detailed examples.",
    "How does the visual cortex process different features like edges, motion, and color?",
    "What is the role of the thalamus in sensory processing across different modalities?"
]

class StressTester:
    def __init__(self):
        self.results = []
        self.session_token = None

    async def login(self, session):
        """Login to get session token if needed"""
        # Adjust login payload based on your auth system
        login_data = {
            "course_code": "PSYC101",  # Adjust to your course code
            "password": "your_password"  # Adjust if needed
        }

        try:
            async with session.post(LOGIN_ENDPOINT, json=login_data) as response:
                if response.status == 200:
                    data = await response.json()
                    self.session_token = data.get('token')
                    print(f"✅ Login successful")
                    return True
                else:
                    print(f"❌ Login failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Login error: {e}")
            return False

    async def send_question(self, session, user_id, question_idx):
        """Send a single question and measure response"""
        question = QUESTIONS[question_idx % len(QUESTIONS)]

        headers = {"Content-Type": "application/json"}
        if self.session_token:
            headers['Authorization'] = f'Bearer {self.session_token}'

        # Fixed payload to match your FastAPI model
        payload = {
            "question": question,
            "temperature": 0.7,
            "system_prompt": None,
            "chat_id": None
        }

        print(f"🔍 Debug: Sending to {CHAT_ENDPOINT} with payload: {payload}")
        start_time = time.time()

        try:
            async with session.post(CHAT_ENDPOINT,
                                  json=payload,
                                  headers=headers,
                                  timeout=aiohttp.ClientTimeout(total=30)) as response:

                end_time = time.time()
                response_time = end_time - start_time

                if response.status == 200:
                    # For streaming endpoint, we need to read the stream
                    response_data = ""
                    async for line in response.content:
                        chunk = line.decode('utf-8').strip()
                        if chunk.startswith('data: '):
                            response_data += chunk[6:]  # Remove 'data: ' prefix
                    
                    result = {
                        'user_id': user_id,
                        'question_idx': question_idx,
                        'response_time': response_time,
                        'status': 'success',
                        'timestamp': datetime.now().isoformat(),
                        'response_length': len(response_data)
                    }
                    print(f"✅ User {user_id}: {response_time:.2f}s ({len(response_data)} chars)")

                elif response.status == 429:
                    result = {
                        'user_id': user_id,
                        'question_idx': question_idx,
                        'response_time': response_time,
                        'status': 'rate_limited',
                        'timestamp': datetime.now().isoformat()
                    }
                    print(f"⚠️  User {user_id}: RATE LIMITED (429)")

                else:
                    # Get error details
                    error_text = await response.text()
                    result = {
                        'user_id': user_id,
                        'question_idx': question_idx,
                        'response_time': response_time,
                        'status': f'error_{response.status}',
                        'timestamp': datetime.now().isoformat(),
                        'error_detail': error_text[:200]  # First 200 chars of error
                    }
                    print(f"❌ User {user_id}: ERROR {response.status} - {error_text[:100]}")

                self.results.append(result)
                return result

        except asyncio.TimeoutError:
            result = {
                'user_id': user_id,
                'question_idx': question_idx,
                'response_time': 30.0,
                'status': 'timeout',
                'timestamp': datetime.now().isoformat()
            }
            print(f"⏰ User {user_id}: TIMEOUT")
            self.results.append(result)
            return result

        except Exception as e:
            result = {
                'user_id': user_id,
                'question_idx': question_idx,
                'response_time': time.time() - start_time,
                'status': f'exception_{type(e).__name__}',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
            print(f"💥 User {user_id}: EXCEPTION {e}")
            self.results.append(result)
            return result

    async def stress_test_round(self, num_users, questions_per_user=3):
        """Run one round of stress testing with specified number of users"""
        print(f"\n🚀 Starting stress test: {num_users} concurrent users, {questions_per_user} questions each")
        print(f"📊 Total requests: {num_users * questions_per_user}")
        print("=" * 60)

        # Use connection limits to prevent overwhelming the server
        connector = aiohttp.TCPConnector(limit=50, limit_per_host=20)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Login first if your system requires it
            # await self.login(session)

            # Create tasks for all users
            tasks = []
            for user_id in range(num_users):
                for question_idx in range(questions_per_user):
                    task = self.send_question(session, user_id, question_idx)
                    tasks.append(task)

            # Execute all requests concurrently
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time

            # Analyze results
            self.analyze_results(num_users, questions_per_user, total_time)

            return results

    def analyze_results(self, num_users, questions_per_user, total_time):
        """Analyze and display test results"""
        total_requests = len(self.results)
        if total_requests == 0:
            return

        # Count by status
        status_counts = {}
        response_times = []

        for result in self.results:
            status = result['status']
            status_counts[status] = status_counts.get(status, 0) + 1
            if result['status'] == 'success':
                response_times.append(result['response_time'])

        # Calculate stats
        success_rate = (status_counts.get('success', 0) / total_requests) * 100
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        requests_per_second = total_requests / total_time

        print(f"\n📈 RESULTS FOR {num_users} CONCURRENT USERS:")
        print(f"   Total requests: {total_requests}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Requests/second: {requests_per_second:.2f}")
        print(f"   Success rate: {success_rate:.1f}%")
        if response_times:
            print(f"   Avg response time: {avg_response_time:.2f}s")
            print(f"   Min response time: {min(response_times):.2f}s")
            print(f"   Max response time: {max(response_times):.2f}s")

        print(f"\n📊 Status breakdown:")
        for status, count in status_counts.items():
            print(f"   {status}: {count} ({count/total_requests*100:.1f}%)")

        # Show error details if any
        error_results = [r for r in self.results if 'error_detail' in r]
        if error_results:
            print(f"\n🔍 Error samples:")
            for error in error_results[:3]:  # Show first 3 errors
                print(f"   {error['status']}: {error.get('error_detail', 'No details')}")

        # Clear results for next round
        self.results = []

async def main():
    """Run progressive stress test"""
    tester = StressTester()

    print("🎯 CHATBOT STRESS TEST STARTING")
    print("📋 Testing Sensation & Perception Tutor Chatbot")
    print("⚡ Looking for rate limits and capacity bounds")
    print("=" * 60)

    # Start with smaller numbers to isolate issues
    test_levels = [1, 2, 3, 5, 8, 10, 15, 20]

    for num_users in test_levels:
        try:
            await tester.stress_test_round(num_users, questions_per_user=2)

            # Brief pause between rounds
            await asyncio.sleep(3)

            # Stop if we hit significant errors
            recent_results = tester.results[-num_users*2:] if tester.results else []
            error_rate = sum(1 for r in recent_results if r.get('status') != 'success') / len(recent_results) if recent_results else 0
            
            if error_rate > 0.5:  # If more than 50% errors
                print(f"\n🛑 HIGH ERROR RATE DETECTED at {num_users} users!")
                print(f"🎯 Error rate: {error_rate*100:.1f}% - This may be your capacity limit.")
                break

        except KeyboardInterrupt:
            print("\n⏹️ Test stopped by user")
            break
        except Exception as e:
            print(f"\n💥 Test failed at {num_users} users: {e}")
            break

    print(f"\n🏁 STRESS TEST COMPLETE!")
    print("💡 Check your monitoring terminals for server resource usage.")
    print("🔧 Run 'docker logs -f your-container-name' to see server-side errors.")

if __name__ == "__main__":
    # Run the stress test
    asyncio.run(main())
