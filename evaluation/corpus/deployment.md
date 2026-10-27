# Mercury deployment

The Mercury production service is deployed in the Beijing region.
The Mercury worker retries transient HTTP 503 responses three times with exponential backoff.
The first retry waits two seconds, and each later retry doubles the delay.
