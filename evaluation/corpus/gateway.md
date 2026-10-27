# Traffic path

The API Gateway routes traffic to the Mercury worker.
After retries are exhausted, the Mercury worker sends failed jobs to dead_letter_queue.
