// Basic k6 load test script for benchmarking
// Tests against a simple HTTP endpoint or generates CPU load
import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  thresholds: {
    http_req_duration: ["p(95)<500"],
  },
};

export default function () {
  // If K6_TARGET_URL is set, test against that URL
  // Otherwise, generate synthetic CPU-bound work
  const targetUrl = __ENV.K6_TARGET_URL || "https://test.k6.io";

  const res = http.get(targetUrl);
  check(res, {
    "status is 200": (r) => r.status === 200,
  });
  sleep(0.1);
}
