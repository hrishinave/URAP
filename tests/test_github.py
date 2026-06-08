import unittest
import warnings

from llm_fp_mining.github import GitHubClient, parse_link_header


warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")


class FakeResponse:
    def __init__(self, status_code, payload, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        return self.responses.pop(0)


class GitHubClientTests(unittest.TestCase):
    def test_parse_link_header(self):
        links = parse_link_header('<https://api.github.com/x?page=2>; rel="next", <https://api.github.com/x?page=3>; rel="last"')

        self.assertEqual(links["next"], "https://api.github.com/x?page=2")
        self.assertEqual(links["last"], "https://api.github.com/x?page=3")

    def test_retries_retry_after_rate_limit(self):
        sleeps = []
        session = FakeSession(
            [
                FakeResponse(403, {"message": "rate limited"}, {"Retry-After": "0"}),
                FakeResponse(200, {"ok": True}, {"X-RateLimit-Remaining": "1"}),
            ]
        )
        client = GitHubClient(token="token", session=session, sleep=sleeps.append)

        data, _ = client.get_json("https://api.github.com/test")

        self.assertEqual(data, {"ok": True})
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(sleeps, [1])


if __name__ == "__main__":
    unittest.main()
